"""Experimental private-validation rules; no transferred convergence guarantees."""
import copy
import hashlib
import math
from dataclasses import replace
import numpy as np
import torch
from .research.contracts import ValidationVote
from .research.defenses import flatten_update
from .research.runner_helpers import _apply_delta

def paired_decision(baseline, candidate, tolerance, standard_errors):
    baseline=np.asarray(baseline,dtype=float)
    candidate=np.asarray(candidate,dtype=float)
    if baseline.ndim!=1 or candidate.ndim!=1 or baseline.shape!=candidate.shape or len(baseline)==0:
        raise ValueError('paired losses require identical nonempty one-dimensional shapes')
    if not np.isfinite(baseline).all() or not np.isfinite(candidate).all():
        raise ValueError('nonfinite paired losses')
    differences=candidate-baseline
    if differences.ndim!=1 or len(differences)==0 or not np.isfinite(differences).all():
        raise ValueError('invalid paired losses')
    mean=float(differences.mean())
    # One example supplies no variance estimate: abstain, represented as no approval.
    if len(differences)<2:return False,{'mean_loss_change':mean,'n':len(differences),'insufficient_samples':True}
    se=float(differences.std(ddof=1)/np.sqrt(len(differences)))
    return bool(mean <= tolerance+standard_errors*se),{'mean_loss_change':mean,'standard_error':se,'n':len(differences)}

def rank_approve(scores,candidate,keep_fraction):
    values=np.asarray(list(scores.values()),dtype=float)
    if not len(values) or not np.isfinite(values).all():raise ValueError('invalid rank scores')
    # Include all threshold ties; do not privilege client IDs on equal scores.
    count=max(1,math.ceil(keep_fraction*len(values)))
    cutoff=float(np.sort(values)[-count])
    return bool(scores[candidate]>=cutoff),cutoff

def clip_updates(updates,ratio):
    norms=[float(flatten_update(u).norm()) for u in updates]
    if not all(math.isfinite(n) for n in norms):raise ValueError('nonfinite update norm')
    median=float(np.median(norms)) if norms else 0.0
    if ratio==0:return updates,{'median_norm':median,'scales':[1.0]*len(updates)}
    bound=ratio*median
    scales=[min(1.0,bound/max(n,1e-12)) for n in norms]
    clipped=[replace(u,state={k:v*scale if v.is_floating_point() else v for k,v in u.state.items()})
             for u,scale in zip(updates,scales)]
    return clipped,{'median_norm':median,'scales':scales,'bound':bound}

class LossCommittee:
    def __init__(self,runner,model,x,y,split,updates):
        self.runner=runner;self.model=model;self.x=x;self.y=y;self.split=split
        self.updates={u.client_id:u for u in updates}
        self.baseline={};self.cache={};self.models={}
        self.mass=sum(u.sample_count for u in updates)
        self.median_norm=max(float(np.median([float(flatten_update(u).norm()) for u in updates])),1e-12) if updates else 1.

    def losses(self,model,indices):
        self.runner.validation_work['private_example_evaluations'] += len(indices)
        self.runner.validation_work['private_forward_passes'] += math.ceil(len(indices)/self.runner.config.evaluation_batch_size)
        if self.y.ndim!=1:raise ValueError('loss rules only support classification')
        model.eval();values=[]
        with torch.no_grad():
            for start in range(0,len(indices),self.runner.config.evaluation_batch_size):
                ids=indices[start:start+self.runner.config.evaluation_batch_size]
                x=torch.as_tensor(self.x[ids],dtype=torch.float32,device=self.runner.device)
                y=torch.as_tensor(self.y[ids],dtype=torch.long,device=self.runner.device)
                values.extend(torch.nn.functional.cross_entropy(model(x),y,reduction='none').cpu().tolist())
        return np.asarray(values)

    def pair(self,validator,client):
        if validator not in self.baseline:
            self.baseline[validator]=self.losses(self.model,self.split.validation[validator])
        key=(validator,client)
        if key not in self.cache:
            if client not in self.models:
                candidate=copy.deepcopy(self.model);u=self.updates[client]
                probe=self.runner.config.validation_variant!='original'
                scale=u.sample_count/self.mass if probe or self.runner.config.vote_rule=='scaled_loss' else 1.
                _apply_delta(candidate,{k:v*scale if v.is_floating_point() else v for k,v in u.state.items()})
                self.models[client]=candidate
            self.cache[key]=self.losses(self.models[client],self.split.validation[validator])
        return self.baseline[validator],self.cache[key]

    def score_mean(self,validator,values):
        if self.runner.config.validation_variant!='macro_probe':return float(np.mean(values))
        labels=self.y[self.split.validation[validator]]
        # Equal mass for observed classes only; absent classes remain unknown.
        return float(np.mean([np.mean(values[labels==label]) for label in np.unique(labels)]))

    def vote(self,validator,client,corrupt,forced=None):
        c=self.runner.config
        baseline,candidate=self.pair(validator,client)
        if c.vote_rule=='rank_loss':
            scores={}
            # Exclude a validator's own proposal from its comparison pool.
            for peer,u in self.updates.items():
                if peer==validator:continue
                b,l=self.pair(validator,peer)
                scale=u.sample_count/self.mass if c.validation_variant!='original' else 1.
                norm=scale*float(flatten_update(u).norm())/self.median_norm
                scores[peer]=self.score_mean(validator,b-l)-c.ranking_norm_penalty*norm**2
            approve,cutoff=rank_approve(scores,client,c.ranking_keep_fraction)
            diagnostic={'loss_score':scores[client],'cutoff':cutoff,'n':len(baseline),
                        'rank_approved':approve,'variant':c.validation_variant,
                        'probe_scale':self.updates[client].sample_count/self.mass if c.validation_variant!='original' else 1.,
                        'observed_classes':int(len(np.unique(self.y[self.split.validation[validator]])))}
            if c.validation_variant=='margin_probe':
                nonharmful,paired=paired_decision(baseline,candidate,c.loss_tolerance,c.loss_standard_errors)
                approve=approve or nonharmful
                diagnostic.update(nonharmful_probe=nonharmful,paired=paired)
        else:
            approve,diagnostic=paired_decision(baseline,candidate,c.loss_tolerance,c.loss_standard_errors)
        if forced is not None:approve=bool(forced)
        diagnostic['abstain'] = bool(diagnostic.get('insufficient_samples',False) and forced is None)
        delta=float(np.mean(baseline-candidate))
        diagnostic.update(local_mean_loss_change=-delta,
                          locally_harmful=bool(delta < 0),
                          relative_rank_rejected=(not diagnostic['rank_approved']) if c.vote_rule=='rank_loss' else None,
                          decision_basis='relative_rank' if c.vote_rule=='rank_loss' else 'paired_loss',
                          probe_scale=self.updates[client].sample_count/self.mass if c.vote_rule=='scaled_loss' or c.validation_variant!='original' else 1.,
                          heuristic_acceptance=True,
                          confidence_guarantee=False,
                          establishes_misconduct=False)
        material=f'{validator}:{client}:{delta:.12f}:{int(approve)}'
        return ValidationVote(validator,client,delta,approve,corrupt,hashlib.sha256(material.encode()).hexdigest()),diagnostic
