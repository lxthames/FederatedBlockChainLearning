"""TRUST v2 matrices; legacy matrices remain available for provenance only."""
from dataclasses import replace
from itertools import product
import hashlib
import json
from .config import ExperimentConfig

SEEDS=(13,29,47,71,101)
DEVELOPMENT_SEEDS=(3,5,7)
METHODS=('fedavg','coordinate_median','multi_krum','fltrust','boba','two_layer')
ATTACKS=('none','sign_flip','alie','model_replacement','low_and_slow')


def cell(prefix, **changes):
    defaults=dict(dataset='fashion_mnist',num_clients=50,participation_rate=0.4,
        rounds=100,local_epochs=1,batch_size=64,learning_rate=0.01,
        malicious_fraction=0.3,dirichlet_alpha=0.5,seed=13,attack='sign_flip',
        defense='two_layer',device='auto',evidence_tier='confirmatory',sampling_policy='bounded')
    defaults.update(changes)
    digest=hashlib.sha256(json.dumps(defaults,sort_keys=True).encode()).hexdigest()[:12]
    return ExperimentConfig(experiment_id=f'trust-{prefix}-{defaults["dataset"]}-{defaults["defense"]}-{defaults["attack"]}-s{defaults["seed"]}-{digest}',**defaults)


class Trust2027Matrix:
    def development(self):
        return [cell('development',dataset=d,defense=m,attack=a,seed=s,
            rounds=20,evidence_tier='development')
            for d,m,a,s in product(('fashion_mnist','cifar10'),('fedavg','two_layer'),ATTACKS,DEVELOPMENT_SEEDS)]

    def core_confirmation(self):
        return [cell('core',dataset=d,defense=m,attack=a,seed=s)
            for d,m,a,s in product(('fashion_mnist','cifar10'),METHODS,ATTACKS,SEEDS)]

    def scale_sweep(self):
        return [cell('scale',num_clients=n,defense=m,attack=a,seed=s)
            for n,m,a,s in product((20,50,80),('fedavg','two_layer'),('model_replacement','low_and_slow'),SEEDS)]

    def heterogeneity_sweep(self):
        return [cell('heterogeneity',dirichlet_alpha=alpha,defense=m,attack=a,seed=s)
            for alpha,m,a,s in product((None,1.0,0.5,0.1),('fedavg','two_layer'),('model_replacement','low_and_slow'),SEEDS)]

    def attacker_fraction_sweep(self):
        return [cell('attackers',malicious_fraction=f,defense=m,attack=a,seed=s)
            for f,m,a,s in product((0.1,0.3,0.5),('fedavg','two_layer'),('model_replacement','low_and_slow'),SEEDS)]

    def identity_committee_sweep(self):
        # OFAT around perfect enrollment and no capture; duplicate attempts stay
        # nonzero so FAR actually exercises the identity gate.
        axes=[dict(identity_false_accept_rate=x) for x in (0,0.05,0.1,0.2)]
        axes += [dict(identity_false_reject_rate=x) for x in (0.05,0.1)]
        axes += [dict(malicious_validator_fraction=x) for x in (0.33,0.5)]
        return [cell('identity',duplicate_identity_fraction=0.3,attack=a,seed=s,**axis)
            for axis,a,s in product(axes,('sign_flip','low_and_slow'),SEEDS)]

    def ablations(self):
        variants=[{},dict(ablation='validation_only'),dict(ablation='no_penalties'),dict(detector='adjacent')]
        return [cell('ablation',attack=a,seed=s,**v)
            for v,a,s in product(variants,('none','sign_flip','low_and_slow'),SEEDS)]

    def calibration(self):
        variants=[dict(relative_drop_threshold=d,slope_threshold=t)
            for d,t in product((0.025,0.05,0.1),(0.0025,0.005,0.01))]
        return [cell('calibration',attack='none',rounds=20,seed=s,evidence_tier='development',**v)
            for v,s in product(variants,DEVELOPMENT_SEEDS)]

    def benign_drift(self):
        return [cell('drift',attack='none',benign_drift_round=30,
            benign_drift_magnitude=d,minimum_trust=t,seed=s)
            for d,t,s in product((0.0,0.1,0.2),(0.3,0.45,0.5),SEEDS)]

    def sift_case_study(self):
        return [cell('sift',dataset='sift_routing',num_clients=n,
            participation_rate=1.0,model=m,defense=d,attack=a,
            rounds=50,evidence_tier='case_study',seed=s)
            for n,m,d,a,s in product((4,8),('emh_dnn','mlp'),
                ('fedavg','fltrust','two_layer'),('none','sign_flip','low_and_slow'),SEEDS)]
