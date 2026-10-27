"""Paper-based FLAME/FLShield and explicitly adapted partial-client detection.

No method uses ClientUpdate.is_malicious. Labels are only class labels supplied
to private validation. Source-to-code differences are in BASELINE_AUDIT.md.
"""
import copy
import math
import warnings
from collections import defaultdict,deque
import numpy as np
import torch
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans
from sklearn.covariance import EllipticEnvelope
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.metrics import silhouette_score
from .quasi_newton import SecantHessian
from .research.contracts import AggregationResult,ClientUpdate
from .research.defenses import flatten_update,_state_from_vector,FedAvgDefense,CoordinateMedianDefense
from .research.runner_helpers import _apply_delta
from .research.splits import stream
from .loss_voting import LossCommittee


def vector(model):
    return np.concatenate([v.detach().cpu().numpy().reshape(-1) for v in model.state_dict().values() if v.is_floating_point()]).astype(np.float64)


def cosine_distances(vectors):
    values=np.asarray(vectors,dtype=np.float64)
    norms=np.linalg.norm(values,axis=1)
    denom=np.maximum(norms[:,None]*norms[None,:],1e-30)
    distances=np.clip(1-values@values.T/denom,0,2)
    both_zero=(norms[:,None]==0)&(norms[None,:]==0)
    distances[both_zero]=0
    distances=(distances+distances.T)/2
    np.fill_diagonal(distances,0)
    return distances


def clip_mean(updates,indices,bound,weighted=False):
    if not indices:raise ValueError('no selected updates')
    vectors=torch.stack([flatten_update(updates[i]).double() for i in indices])
    norms=vectors.norm(dim=1)
    vectors*=torch.clamp(bound/norms.clamp_min(1e-30),max=1)[:,None]
    weights=torch.tensor([updates[i].sample_count if weighted else 1 for i in indices],dtype=torch.float64)
    return (vectors*weights[:,None]).sum(0)/weights.sum()


def flame(model,updates,seed,rnd,noise_multiplier=.001):
    if len(updates)<3:raise ValueError('FLAME requires at least three updates')
    deltas=np.stack([flatten_update(u).numpy() for u in updates]).astype(np.float64)
    # Paper Algorithm 1 clusters full local models, NOT update directions.
    distances=cosine_distances(deltas+vector(model)[None,:])
    clusterer=HDBSCAN(metric='precomputed',min_cluster_size=len(updates)//2+1,
                      min_samples=1,allow_single_cluster=True,core_dist_n_jobs=1)
    labels=clusterer.fit_predict(distances)
    selected=np.flatnonzero(labels==0).tolist()
    if len(selected)<=len(updates)//2:
        raise ValueError('FLAME has no strict-majority cluster; unsupported round')
    bound=float(np.median(np.linalg.norm(deltas,axis=1)))
    result=clip_mean(updates,selected,bound)
    # Separate RNG: defense noise cannot change future client/attack randomness.
    noise=stream(seed,'flame-noise',rnd).normal(0,noise_multiplier*bound,len(result))
    result+=torch.from_numpy(noise)
    return AggregationResult(_state_from_vector(result,updates[0]),
        tuple(updates[i].client_id for i in selected),
        tuple(u.client_id for i,u in enumerate(updates) if i not in selected),
        {'method':'flame_paper','clip_bound':bound,'noise_std':noise_multiplier*bound,
         'cluster_labels':labels.tolist(),'admitted':len(selected)})


def choose_clusters(deltas,seed):
    # KMeans distances/silhouette are computed on an exact low-rank coordinate
    # representation of these n vectors; no lossy random projection or PCA.
    gram=deltas@deltas.T
    eigenvalues,eigenvectors=np.linalg.eigh((gram+gram.T)/2)
    embedding=eigenvectors*np.sqrt(np.maximum(eigenvalues,0))[None,:]
    best=None
    for k in range(2,len(deltas)//2+1):
        labels=KMeans(n_clusters=k,random_state=seed,n_init=10).fit_predict(embedding)
        if len(np.unique(labels))!=k:continue
        score=float(silhouette_score(embedding,labels))
        if best is None or score>best[0]:best=(score,labels)
    if best is None:raise ValueError('FLShield cannot find supported nondegenerate clusters')
    return best


def filter_reports(reports,seed,contamination=.1):
    """reports[v,representative,class]; never impute an entirely unseen class."""
    reports=np.asarray(reports,dtype=float)
    if reports.ndim!=3 or np.isinf(reports).any():raise ValueError('invalid validation reports')
    if np.isnan(reports).all(axis=0).any():
        raise ValueError('FLShield unsupported: class has no qualified validator')
    flat=reports.reshape(len(reports),-1)
    imputed=IterativeImputer(random_state=seed,max_iter=10,sample_posterior=False,
                            skip_complete=True).fit_transform(flat)
    if imputed.shape!=flat.shape or not np.isfinite(imputed).all():raise ValueError('imputation failed')
    captured=[]
    if np.max(np.ptp(imputed,axis=0))<1e-12:
        keep=np.ones(len(reports),dtype=bool)
    else:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            keep=EllipticEnvelope(contamination=contamination,support_fraction=.75,
                random_state=seed).fit_predict(imputed)==1
            captured=[str(w.message) for w in caught]
    if not keep.any():raise ValueError('no retained validation reports')
    scores=imputed.reshape(reports.shape)[keep].mean(0).min(1)
    return scores,keep,captured


class FLShield:
    def __init__(self,runner,x,y,split):
        self.runner=runner;self.x=x;self.y=y;self.split=split
        self.minimum=runner.config.shield_min_class_samples
        self.maximum=runner.config.shield_max_class_samples
        self.classes=runner.config.num_classes
        self.qualified=[]
        for client,ids in enumerate(split.validation):
            groups=[]
            for label in range(self.classes):
                ids_cls=ids[y[ids]==label]
                groups.append(ids_cls if len(ids_cls)>self.minimum else np.array([],dtype=int))
            self.qualified.append(groups)
        support=np.array([[len(g)>0 for g in groups] for groups in self.qualified])
        if not support.any(0).all():raise ValueError('FLShield class coverage assumption unsupported')

    def aggregate(self,model,updates,rnd):
        c=self.runner.config
        deltas=np.stack([flatten_update(u).numpy() for u in updates]).astype(np.float64)
        silhouette,labels=choose_clusters(deltas,int(stream(c.seed,'shield-clusters',rnd).integers(2**31)))
        clusters=[np.flatnonzero(labels==k).tolist() for k in sorted(set(labels))]
        representatives=[]
        for cluster in clusters:
            mean=torch.from_numpy(deltas[cluster].mean(0))
            candidate=copy.deepcopy(model)
            _apply_delta(candidate,_state_from_vector(mean,updates[0]));representatives.append(candidate)
        evaluator=LossCommittee(self.runner,model,self.x,self.y,self.split,[])
        reports=np.full((len(self.qualified),len(clusters),self.classes),np.nan)
        evaluated=0
        for validator,groups in enumerate(self.qualified):
            for label,ids in enumerate(groups):
                if not len(ids):continue
                ids=stream(c.seed,'shield-examples',rnd,validator,label).permutation(ids)[:self.maximum]
                base=evaluator.losses(model,ids)
                for j,candidate in enumerate(representatives):
                    reports[validator,j,label]=float(np.mean(base-evaluator.losses(candidate,ids)))
                    evaluated+=len(ids)
        scores,keep,captured=filter_reports(reports,int(stream(c.seed,'shield-report-filter',rnd).integers(2**31)),c.shield_contamination)
        count=max(1,math.ceil(len(clusters)/2))
        # Equal scores all survive; no arbitrary client-ID preference.
        cutoff=float(np.sort(scores)[-count]);accepted_clusters=np.flatnonzero(scores>=cutoff)
        selected=[i for k in accepted_clusters for i in clusters[k]]
        bound=float(np.median(np.linalg.norm(deltas,axis=1)))
        result=clip_mean(updates,selected,bound)
        floating_bytes=sum(v.numel()*v.element_size() for v in model.state_dict().values())
        return AggregationResult(_state_from_vector(result,updates[0]),tuple(updates[i].client_id for i in selected),
            tuple(u.client_id for i,u in enumerate(updates) if i not in selected),
            {'method':'flshield_cluster_paper_based','clusters':clusters,'silhouette':silhouette,
             'class_min_loss_impacts':scores.tolist(),'retained_validators':np.flatnonzero(keep).tolist(),
             'missing_report_fraction':float(np.isnan(reports).mean()),'covariance_warnings':captured,
             'private_example_evaluations':evaluated,'clip_bound':bound,
             'extra_transmitted_bytes':len(clusters)*len(self.qualified)*floating_bytes+reports.size*8})


def gap_two_clusters(scores,rng,references=10):
    """Tibshirani one-SE gap selection restricted to k=1,2, as disclosed."""
    values=np.asarray(scores,dtype=float).reshape(-1,1)
    if len(values)<6 or np.ptp(values)==0:return np.zeros(len(values),dtype=bool),{'clusters':1}
    seed=int(rng.integers(2**31));models=[KMeans(n_clusters=k,random_state=seed,n_init=10).fit(values) for k in (1,2)]
    references_inertia=[]
    for _ in range(references):
        ref=rng.uniform(values.min(),values.max(),values.shape)
        references_inertia.append([math.log(max(KMeans(n_clusters=k,random_state=seed,n_init=5).fit(ref).inertia_,1e-30)) for k in (1,2)])
    reference=np.array(references_inertia)
    gap=reference.mean(0)-np.log([max(m.inertia_,1e-30) for m in models])
    uncertainty=reference.std(0,ddof=1)*math.sqrt(1+1/references)
    if gap[0]>=gap[1]-uncertainty[1]:return np.zeros(len(values),dtype=bool),{'clusters':1,'gap':gap.tolist()}
    labels=models[1].labels_;means=[float(values[labels==k].mean()) for k in (0,1)]
    return labels==int(np.argmax(means)),{'clusters':2,'gap':gap.tolist(),'cluster_means':means}


class PartialFLDetector:
    """Adaptation: per-client secants across participation gaps, three strikes.

    This is intentionally named fldetector_pp, not a faithful full-participation
    FLDetector reproduction. No restart is implied by an exclusion decision.
    """
    def __init__(self,seed):
        self.seed=seed;self.previous={};self.hessians=defaultdict(SecantHessian)
        self.scores=defaultdict(lambda:deque(maxlen=5));self.strikes=defaultdict(int)
        self.quarantined=set()

    def observe(self,model,updates,rnd):
        current=vector(model);residuals={};curvature={}
        for u in updates:
            client=u.client_id
            if client in self.quarantined:continue
            grad=-flatten_update(u).numpy().astype(np.float64)
            if client in self.previous:
                last_w,last_g=self.previous[client];hessian=self.hessians[client]
                # Predict before admitting the current observation to the history.
                if hessian.pairs:
                    predicted=last_g+hessian.product(current-last_w)
                    residuals[client]=float(np.linalg.norm(predicted-grad))
                hessian.add(current-last_w,grad-last_g)
                curvature[client]=len(hessian.pairs)
            self.previous[client]=(current.copy(),grad.copy())
        scale=max(sum(residuals.values()),1e-30)
        for client,value in residuals.items():self.scores[client].append(value/scale)
        mature=[client for client in residuals if len(self.scores[client])>=3]
        new=[];diagnostic={'clusters':1}
        if len(mature)>=6:
            flag,diagnostic=gap_two_clusters([np.mean(self.scores[i]) for i in mature],stream(self.seed,'detector-gap',rnd))
            for client,high in zip(mature,flag):
                self.strikes[client]=self.strikes[client]+1 if high else 0
                if self.strikes[client]>=3:new.append(client)
        self.quarantined.update(new)
        return {'method':'fldetector_pp_adaptation','new_quarantines':sorted(new),
                'quarantined':sorted(self.quarantined),'residuals':residuals,
                'mature_observed_clients':mature,'valid_secants':curvature,'gap':diagnostic}


class LiteratureDefense:
    def __init__(self,runner,x,y,split):
        self.runner=runner
        self.shield=FLShield(runner,x,y,split) if runner.config.defense=='flshield_cluster' else None
        self.detector=PartialFLDetector(runner.config.seed) if runner.config.defense=='fldetector_pp' else None

    def aggregate(self,model,updates,rnd):
        c=self.runner.config
        if c.defense=='flame':return flame(model,updates,c.seed,rnd,c.flame_noise_multiplier)
        if self.shield:return self.shield.aggregate(model,updates,rnd)
        diagnostic=self.detector.observe(model,updates,rnd)
        admitted=[u for u in updates if u.client_id not in self.detector.quarantined]
        if not admitted:raise ValueError('detector excluded every participant')
        # Median is the prespecified post-detection robust aggregator.
        result=CoordinateMedianDefense().aggregate(model.state_dict(),admitted,None)
        admitted_ids={u.client_id for u in admitted}
        return AggregationResult(result.state,tuple(u.client_id for u in admitted),
            tuple(u.client_id for u in updates if u.client_id not in admitted_ids),diagnostic)
