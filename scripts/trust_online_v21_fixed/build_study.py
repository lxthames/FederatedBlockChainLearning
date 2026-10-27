"""Generate a separate fixed coordinator; preserve the original frozen study."""
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def rendered_study():
    value=(ROOT/'scripts/trust_online_v21/study.py').read_text(encoding='utf-8')
    replacements=[
        ('from .runner import BenchmarkRunner','from scripts.trust_online_v21.runner import BenchmarkRunner'),
        ('from .build_runner import rendered_runner','from scripts.trust_online_v21.build_runner import rendered_runner'),
        ('from .online import OnlineRepair,DeploymentPolicy','from scripts.trust_online_v21.online import OnlineRepair,DeploymentPolicy'),
        ('from .analysis import SEEDS,ATTACKS,ARMS,clean_gate,qualify,extract',
         'from scripts.trust_online_v21.analysis import SEEDS,ATTACKS,ARMS,clean_gate,qualify,extract\nfrom scripts.trust_online_v21 import study as initial\nfrom .parity import historical_weights'),
        ("JOB=ROOT/'artifacts/online-v21'","JOB=ROOT/'artifacts/online-v21-fixed'"),
        ("OUT=ROOT/'results/online-v21'","OUT=ROOT/'results/online-v21-fixed'"),
        ("POLICY_FILE=ROOT/'docs/research/ONLINE_V21.md'","POLICY_FILE=ROOT/'docs/research/ONLINE_V21_FIXED.md'"),
        ("TEST_FILE=ROOT/'tests/test_online_v21.py'","TEST_FILE=ROOT/'tests/test_online_v21_fixed.py'"),
        ("LAUNCHER=ROOT/'scripts/run_online_v21.ps1'","LAUNCHER=ROOT/'scripts/run_online_v21_fixed.ps1'"),
        ("experiment_id=f'v21-femnist-","experiment_id=f'v21fixed-femnist-"),
        ("    v20.assert_frozen()", "    if json.loads(json.dumps(initial.inputs()))!=read(initial.JOB/'frozen-inputs.json'):\n        raise ValueError('original frozen V21 source changed')\n    v20.assert_frozen()"),
        ("(Path(__file__).parent/'runner.py')", "(ROOT/'scripts/trust_online_v21/runner.py')"),
        ("        parent_freeze_sha256=", "        initial_freeze_sha256=digest(initial.JOB/'frozen-inputs.json'),\n        failure_audit_sha256=digest(ROOT/'docs/research/ONLINE_V21_FAILURE_AUDIT.md'),\n        parent_freeze_sha256="),
        ("                    'accepted_malicious_weight_history','validation_work'):", "                    'validation_work'):"),
        ("    return dict(passed=True,checked_models=expected,full_trajectory=expected==101,", "    reconstructed=historical_weights(reference,right)\n    if expected==101 and left['accepted_malicious_weight_history']!=reconstructed:\n        raise ValueError('baseline reconstructed malicious weight mismatch')\n    return dict(passed=True,checked_models=expected,full_trajectory=expected==101,\n                historical_weight_source='accepted IDs and local sample-count events',"),
        ("'scripts.trust_online_v21.study'", "'scripts.trust_online_v21_fixed.study'"),
        ('# V21 online FEMNIST development study','# V21-fixed online FEMNIST development study'),
        ('Frozen V21:', 'Frozen V21-fixed:'),
    ]
    for old,new in replacements:
        if value.count(old)!=1:raise ValueError(f'unexpected original study insertion: {old}')
        value=value.replace(old,new)
    return value


if __name__=='__main__':
    (Path(__file__).parent/'study.py').write_text(rendered_study(),encoding='utf-8')
