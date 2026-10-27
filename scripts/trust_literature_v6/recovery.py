"""Conditional recovery diagnostics, isolated from end-to-end defenses.

fedrecover_partial adapts gradient recovery to local SGD deltas and partial
participation; crab_fixedwindow adapts storage/window/influence definitions.
Neither transfers a certification theorem. Oracle rows are upper bounds only.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from .quasi_newton import SecantHessian
from .literature import PartialFLDetector,vector
from .research.contracts import ClientUpdate
from .research.defenses import FedAvgDefense,flatten_update,_state_from_vector
from .research.runner_helpers import _apply_delta,_state_delta,_attack
from .research.attacks import AttackContext,LowAndSlowAttack
from .research.runner_v2 import BenchmarkRunner,model_hash
from .research.splits import make_splits,stream
from .femnist import digest,dump


def load_row(source,rnd):
    return torch.load(Path(source)/'offchain-history'/f'round-{rnd:03}.pt',map_location='cpu',weights_only=True)


def updates_from(row):
    return [ClientUpdate(u['client_id'],u['state'],u['sample_count']) for u in row['updates']]


def detect_archive(source,model_factory,seed,rounds):
    detector=PartialFLDetector(seed);reports=[];model=model_factory()
    for rnd in range(1,rounds+1):
        row=load_row(source,rnd);model.load_state_dict(row['model_state'])
        reports.append(dict(round=rnd,**detector.observe(model,updates_from(row),rnd)))
    return sorted(detector.quarantined),reports


def weighted_vector(updates):
    if not updates:raise ValueError('no clients remain for aggregation')
    mass=sum(u.sample_count for u in updates)
    return sum(flatten_update(u).double().numpy()*u.sample_count/mass for u in updates)


def recovery_binding(source,c,method,suspects,identity_role):
    source=Path(source)
    return {'config':c.to_dict(),'method':method,'suspects':sorted(suspects),'identity_role':identity_role,
            'source_trace_sha256':digest(source/'chain-trace.json'),
            'source_summary_sha256':digest(source/'summary.json'),
            'history_fingerprints':{p.name:digest(p) for p in sorted((source/'offchain-history').glob('*.pt'))}}


def selective_plan(source,model_factory,x_reference,suspects,rounds,device):
    """Explicit Crab adaptation: fixed five-round KL windows and mean influence."""
    model=model_factory().to(device)
    reference=torch.as_tensor(x_reference,dtype=torch.float32,device=device)
    if len(reference)==0:raise ValueError('selective recovery requires disjoint reference probes')
    def probabilities(state):
        model.load_state_dict(state);model.eval()
        with torch.no_grad():return model(reference).softmax(1).clamp_min(1e-12).cpu().numpy()
    scored=[];previous=None
    for rnd in range(1,rounds+1):
        row=load_row(source,rnd)
        if previous is None:previous=probabilities(row['model_state'])
        after=load_row(source,rnd+1)['model_state'] if rnd<rounds else torch.load(Path(source)/'offchain-history/final.pt',weights_only=True)
        current=probabilities(after)
        kl=float(np.mean(np.sum(current*np.log(current/previous),axis=1)))
        previous=current
        updates=updates_from(row);mean=weighted_vector(updates)
        norms=np.array([float(flatten_update(u).norm()) for u in updates])
        cosine=np.array([float(flatten_update(u).double().numpy()@mean)/max(norm*np.linalg.norm(mean),1e-30) for u,norm in zip(updates,norms)])
        count=max(1,math.ceil(.7*len(updates)));selected=np.argsort(-cosine,kind='stable')[:count].tolist()
        scored.append({'round':rnd,'kl':kl,'clients':[updates[i].client_id for i in selected]})
    retained=[]
    for start in range(0,len(scored),5):
        window=scored[start:start+5];keep=max(1,math.ceil(.6*len(window)))
        retained.extend(sorted(window,key=lambda r:(-r['kl'],r['round']))[:keep])
    retained.sort(key=lambda r:r['round'])
    sensitivity=benign_influence=0.;checkpoint=0;reports=[]
    for r in retained:
        updates=[u for u in updates_from(load_row(source,r['round'])) if u.client_id in r['clients']]
        remaining=[u for u in updates if u.client_id not in suspects]
        if not remaining:
            reports.append(dict(round=r['round'],reason='no unquarantined retained clients'));continue
        all_mean=weighted_vector(updates);good_mean=weighted_vector(remaining)
        sensitivity+=float(np.linalg.norm(all_mean-good_mean))
        benign_influence+=float(np.linalg.norm(good_mean))
        qualifies=sensitivity<=.3*benign_influence
        if qualifies:checkpoint=r['round']
        reports.append(dict(round=r['round'],sensitivity=sensitivity,threshold=.3*benign_influence,qualifies=qualifies))
    logical_bytes=sum((Path(source)/'offchain-history'/f"round-{r['round']:03}.pt").stat().st_size for r in retained)
    return checkpoint,retained,reports,logical_bytes


class RecoveryPilot(BenchmarkRunner):
    def request(self,model,x,y,split,clients,rnd,attackers):
        """Simulator oracle implements actual client behavior; defense sees no labels."""
        c=self.config;updates=[];active=c.attack!='none' and rnd>=c.attack_start_round
        for client in clients:
            local=copy.deepcopy(model);ids=split.train[client]
            self._train(local,x[ids],y[ids],rnd,client,
                        poison=active and client in attackers and c.attack=='model_replacement')
            updates.append(ClientUpdate(client,_state_delta(local,model),len(ids),client in attackers))
        bad=sum(u.is_malicious for u in updates) if active else 0
        if not bad:return updates
        attack=LowAndSlowAttack(c.gradual_max_norm_ratio,c.gradual_ramp_rounds) if c.attack=='low_and_slow' else _attack(c.attack)
        honest=tuple(u for u in updates if not u.is_malicious)
        return [attack.apply(u,AttackContext(rnd-c.attack_start_round+1,c.seed,len(updates),bad,honest))
                if active and u.is_malicious else u for u in updates]

    def recover(self,source,method,suspects,identity_role,x,y,ex,ey,model_factory):
        c=self.config;source=Path(source);start=time.perf_counter()
        if method not in {'retrain','fedrecover_partial','crab_fixedwindow'}:raise ValueError('unknown recovery')
        suspects=set(suspects)
        baseline=json.loads((source/'summary.json').read_text())
        source_trace=digest(source/'chain-trace.json')
        provenance={'source_experiment':baseline['experiment_id'],'source_trace_sha256':source_trace,
                    'identity_role':identity_role,'suspects':sorted(suspects),'method':method}
        binding=recovery_binding(source,c,method,suspects,identity_role)
        dump(self.output_dir/'configuration.json',binding)
        if not suspects:
            result={**provenance,'status':'unsupported_no_detected_suspects','publication_ready':False}
            dump(self.output_dir/'summary.json',result);return result
        split=make_splits(y,c,external_test=True)
        attackers=set(stream(c.seed,'poisoning-identities').choice(c.num_clients,math.floor(c.num_clients*c.malicious_fraction),replace=False).tolist())
        # Attack identities are used below ONLY by the simulator client callback
        # and for final diagnostic attribution counts, never for checkpoint choice.
        model=model_factory().to(self.device);model.load_state_dict(load_row(source,1)['model_state'])
        checkpoint=0;plan=None;selection=[];logical_bytes=0
        if method=='crab_fixedwindow':
            checkpoint,plan,selection,logical_bytes=selective_plan(source,model_factory,x[split.reference],suspects,c.rounds,self.device)
            state=load_row(source,checkpoint+1)['model_state'] if checkpoint<c.rounds else torch.load(source/'offchain-history/final.pt',weights_only=True)
            model.load_state_dict(state)
        steps=list(range(1,c.rounds+1)) if plan is None else [r['round'] for r in plan if r['round']>checkpoint]
        by_round={} if plan is None else {r['round']:r['clients'] for r in plan}
        hessians={};exact_requests=estimated_requests=0;extra_bytes=0;fallbacks=0
        trace=[{'kind':'recovery_start','model_hash':model_hash(model),'checkpoint':checkpoint,
                'source_trace_sha256':source_trace}]
        for position,rnd in enumerate(steps):
            row=load_row(source,rnd);original={u.client_id:u for u in updates_from(row)}
            clients=[i for i in original if i not in suspects and (plan is None or i in by_round[rnd])]
            if not clients:raise ValueError('recovery round has no remaining participants')
            original_model=model_factory();original_model.load_state_dict(row['model_state'])
            offset=vector(model)-vector(original_model)
            exact=(method!='fedrecover_partial' or position<6 or (position-6+1)%5==0 or position>=len(steps)-3)
            pending=[];estimates={}
            for client in clients:
                hessian=hessians.setdefault(client,SecantHessian(memory=3))
                if exact or not hessian.pairs:pending.append(client);continue
                orig=flatten_update(original[client]).double().numpy()
                estimate=orig-hessian.product(offset)  # gradients are -SGD deltas
                threshold=max(float(np.max(np.abs(orig)))*2,1e-6)
                if not np.isfinite(estimate).all() or float(np.max(np.abs(estimate)))>threshold:
                    pending.append(client);fallbacks+=1
                else:
                    estimates[client]=ClientUpdate(client,_state_from_vector(torch.from_numpy(estimate),original[client]),original[client].sample_count)
                    estimated_requests+=1
            actual=self.request(model,x,y,split,pending,rnd,attackers) if pending else []
            exact_requests+=len(actual)
            for u in actual:
                estimates[u.client_id]=u
                if method=='fedrecover_partial':
                    difference=-(flatten_update(u)-flatten_update(original[u.client_id])).double().numpy()
                    hessians[u.client_id].add(offset,difference)
            selected=[estimates[i] for i in clients]
            if method=='crab_fixedwindow':
                calibrated=[]
                for u in selected:
                    old_norm=float(flatten_update(original[u.client_id]).norm());new_norm=float(flatten_update(u).norm())
                    scale=old_norm/max(new_norm,1e-30)
                    calibrated.append(ClientUpdate(u.client_id,{k:v*scale if v.is_floating_point() else v for k,v in u.state.items()},u.sample_count))
                selected=calibrated
            result=FedAvgDefense().aggregate(model.state_dict(),selected,None);_apply_delta(model,dict(result.state))
            width=sum(v.numel()*v.element_size() for v in model.state_dict().values())
            extra_bytes+=2*width*len(actual)
            record={'kind':'recovery_step','step':position+1,'source_round':rnd,'participants':clients,
                    'exact_clients':pending,'estimated_clients':[i for i in clients if i not in pending],
                    'model_hash':model_hash(model)}
            trace.append(record);self.logger.log('recovery_step',rnd,record)
        # Held-out development labels are first used after recovery decisions.
        utility=self._score(model,ex,ey)
        mask=ey!=c.backdoor_target_label
        asr=self._score(model,self._apply_trigger(ex[mask]),np.full(int(mask.sum()),c.backdoor_target_label,dtype=np.int64)) if mask.any() else None
        history_files=list((source/'offchain-history').glob('*.pt'))
        report={**provenance,'status':'completed','publication_ready':False,'checkpoint':checkpoint,
            'steps':len(steps),'final_utility':utility,'final_trigger_target_rate':asr,
            'utility_change_vs_source':utility-baseline['final_utility'],
            'exact_client_requests':exact_requests,'estimated_client_requests':estimated_requests,
            'abnormality_fallbacks':fallbacks,'invalid_curvature_pairs':sum(h.rejected for h in hessians.values()),
            'extra_transmitted_bytes':extra_bytes,'runtime_seconds':time.perf_counter()-start,
            'full_shared_history_bytes':sum(p.stat().st_size for p in history_files),
            'retained_round_file_bytes_upper_bound':logical_bytes if plan is not None else None,
            'honest_clients_quarantined':len(suspects-attackers),'malicious_clients_quarantined':len(suspects&attackers),
            'remaining_malicious_clients':len(attackers-suspects),
            'source_config_sha256':digest(source/'config.json'),
            'history_fingerprints':{p.name:digest(p) for p in history_files}}
        dump(self.output_dir/'selection.json',{'plan':plan,'sensitivity_reports':selection})
        dump(self.output_dir/'recovery-trace.json',trace);dump(self.output_dir/'summary.json',report)
        return report
