"""Trigger-agnostic diagnostic challenges; never read attack identities or eval data."""
import copy
import math
import numpy as np
import torch
from .research.contracts import AggregationResult, ClientUpdate
from .research.runner_helpers import _apply_delta, _update_bytes
from .research.splits import stream
from .loss_voting import paired_decision

CAP = 16
QUANTILE = .995
ARMS = ('current', 'perturbation', 'aggregate', 'combined', 'clip_only')


def challenge_bank(images, seed, rnd, validator):
    if images.ndim != 4 or images.shape[1] != 1:
        raise ValueError('V14 challenges require grayscale NCHW images')
    rng = stream(seed, 'challenge-bank-v14', rnd, validator)
    bank = [images.copy()]
    h, w = images.shape[-2:]
    # All positions are eligible; no attack configuration is inspected.
    for size, value in ((2, 0.), (2, 1.), (4, .5)):
        modified = images.copy()
        top, left = int(rng.integers(h-size+1)), int(rng.integers(w-size+1))
        modified[:, :, top:top+size, left:left+size] = value
        bank.append(modified)
    bank.append(np.clip(images+rng.normal(0., .08, images.shape).astype('float32'), 0., 1.))
    return np.concatenate(bank).astype('float32')


def collapse_score(parent, candidate):
    """Positive change in perturbation-induced target concentration, max over classes/bank."""
    if parent.shape != candidate.shape or parent.ndim != 3 or parent.shape[0] != 5:
        raise ValueError('expected five paired probability blocks')
    contrast = (candidate[1:]-candidate[0]).mean(axis=1) - (parent[1:]-parent[0]).mean(axis=1)
    return max(0., float(contrast.max()))


def passes(score, threshold):
    if not math.isfinite(score) or not math.isfinite(threshold) or threshold < 0:
        raise ValueError('invalid challenge decision input')
    return score <= threshold


class ChallengeRound:
    def __init__(self, runner, parent, x, y, split, rnd, updates, eligible):
        self.runner, self.parent, self.x, self.y = runner, parent, x, y
        self.split, self.rnd, self.updates, self.eligible = split, rnd, updates, eligible
        self.arm = runner.challenge_arm
        if self.arm not in ARMS+('calibrate',):raise ValueError('unknown challenge arm')
        self.cache, self.models = {}, {}
        self.extra_transmitted_bytes = 0
        self.mass = sum(u.sample_count for u in updates)
        if self.arm not in ('current','clip_only','calibrate'):
            t = runner.challenge_thresholds
            if not t or any(k not in t or not math.isfinite(t[k]) for k in ('candidate','aggregate')):
                raise ValueError('defense requires frozen clean calibration')

    def probabilities(self, model, bank):
        model.eval()
        work=self.runner.validation_work
        work['challenge_example_evaluations']=work.get('challenge_example_evaluations',0)+len(bank)
        work['challenge_forward_passes']=work.get('challenge_forward_passes',0)+math.ceil(len(bank)/self.runner.config.evaluation_batch_size)
        pieces=[]
        with torch.no_grad():
            for start in range(0,len(bank),self.runner.config.evaluation_batch_size):
                bx=torch.as_tensor(bank[start:start+self.runner.config.evaluation_batch_size],device=self.runner.device)
                pieces.append(model(bx).softmax(1).cpu().numpy())
        return np.concatenate(pieces).reshape(5,len(bank)//5,-1)

    def baseline(self, validator):
        if validator not in self.cache:
            ids=self.split.validation[validator]
            rng=stream(self.runner.config.seed,'challenge-examples-v14',self.rnd,validator)
            ids=np.asarray(ids)[rng.permutation(len(ids))[:CAP]]
            if len(ids)<2:raise ValueError('challenge validation needs at least two examples')
            bank=challenge_bank(self.x[ids],self.runner.config.seed,self.rnd,validator)
            self.cache[validator]=(bank,self.probabilities(self.parent,bank),self.y[ids])
        return self.cache[validator]

    def evaluate(self, model, validator, stage, client=None):
        bank,parent,labels=self.baseline(validator)
        candidate=self.probabilities(model,bank)
        score=collapse_score(parent,candidate)
        ids=np.arange(len(labels)); tiny=np.finfo('float32').tiny
        before=-np.log(np.maximum(parent[0,ids,labels],tiny))
        after=-np.log(np.maximum(candidate[0,ids,labels],tiny))
        clean_ok,clean=paired_decision(before,after,0.,2.)
        threshold=None if self.arm=='calibrate' else self.runner.challenge_thresholds[stage]
        approve=True if threshold is None else passes(score,threshold)
        self.runner.logger.log('challenge_vote',self.rnd,dict(stage=stage,client=client,validator=validator,
            score=score,threshold=threshold,challenge_approve=approve,clean_approve=clean_ok,
            clean_loss=clean,examples=len(labels),arm=self.arm))
        return approve,clean_ok

    def candidate(self, validator, client):
        if self.arm not in ('calibrate','perturbation','combined'):return True
        if client not in self.models:
            u=next(u for u in self.updates if u.client_id==client)
            model=copy.deepcopy(self.parent)
            _apply_delta(model,{k:v*(u.sample_count/self.mass) if v.is_floating_point() else v for k,v in u.state.items()})
            self.models[client]=model
        ok,_=self.evaluate(self.models[client],validator,'candidate',client)
        return ok

    def aggregate(self, result):
        if self.arm not in ('calibrate','aggregate','combined') or not result.accepted_client_ids:return result,False
        model=copy.deepcopy(self.parent);_apply_delta(model,dict(result.state))
        size=min(15,len(self.eligible)); quorum=math.ceil(2*size/3)
        if size<3:raise ValueError('aggregate committee too small')
        members=sorted(stream(self.runner.config.seed,'aggregate-committee-v14',self.rnd).choice(self.eligible,size,replace=False).tolist())
        self.extra_transmitted_bytes += (_update_bytes(ClientUpdate(-1,result.state,0))+96)*len(members)
        votes=[self.evaluate(model,v,'aggregate') for v in members]
        yes=sum(challenge and clean for challenge,clean in votes)
        blocked=self.arm!='calibrate' and yes<quorum
        self.runner.logger.log('aggregate_challenge',self.rnd,dict(members=members,quorum=quorum,
            affirmative=yes,blocked=blocked,proposed_accepted=list(result.accepted_client_ids)))
        if blocked:
            return AggregationResult({k:torch.zeros_like(v) for k,v in result.state.items()},(),
                tuple(sorted(set(result.accepted_client_ids)|set(result.rejected_client_ids))),
                dict(result.diagnostics,aggregate_challenge_blocked=True)),True
        return result,False
