"""Explicit bindings for diagnostic interventions; old summaries fail closed."""
import hashlib
import json


def digest_value(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def split_bindings(split):
    groups = lambda x: [g.tolist() for g in x]
    return {
        'train': digest_value(groups(split.train)),
        'reference': digest_value(split.reference.tolist()),
        'evaluation_groups': digest_value(groups(split.evaluation_groups)),
        'original_validation': digest_value(groups(split.original_validation)),
        'validation': digest_value(groups(split.validation)),
        'validation_pool': digest_value(sorted(i for g in split.validation for i in g.tolist())),
        'validation_counts': [len(g) for g in split.validation],
        'intervention_verified': True,
    }


TRAINING_FIELDS = ('dataset', 'seed', 'num_clients', 'participation_rate', 'rounds',
    'local_epochs', 'batch_size', 'learning_rate', 'optimizer_momentum', 'num_classes',
    'model', 'sampling_policy', 'malicious_fraction', 'attack', 'attack_start_round',
    'backdoor_fraction', 'backdoor_target_label', 'benign_drift_round', 'benign_drift_magnitude')
TRAINING_FIELDS += ('local_training_budget','local_steps')


def compare_pair(left, right, *, allow_validation_redistribution=False):
    """Allow the declared intervention only; report missing evidence explicitly."""
    reasons = []
    for key in ('participants', 'data_fingerprint', 'test_fingerprint', 'initial_model_hash',
                'training_binding', 'source_hash'):
        if key not in left or key not in right or left[key] != right[key]:
            reasons.append(key)
    a, b = left.get('split_bindings', {}), right.get('split_bindings', {})
    for key in ('train', 'reference', 'evaluation_groups', 'original_validation',
                'validation_pool', 'validation_counts'):
        if key not in a or key not in b or a[key] != b[key]:
            reasons.append('split.'+key)
    if a.get('intervention_verified') is not True or b.get('intervention_verified') is not True:
        reasons.append('split.intervention_verified')
    if 'validation' not in a or 'validation' not in b:
        reasons.append('split.validation')
    elif a['validation'] != b['validation']:
        distributions = {left.get('configuration', {}).get('validation_distribution'),
                         right.get('configuration', {}).get('validation_distribution')}
        if not allow_validation_redistribution or distributions != {'private', 'redistributed'}:
            reasons.append('undeclared_validation_change')
    return {'matched': not reasons, 'mismatches': reasons,
            'validation_redistribution_allowed': allow_validation_redistribution}
