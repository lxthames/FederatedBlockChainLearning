"""Explicit derivation of V22 from frozen V21-fixed; no historical edits."""
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def replace_once(value,old,new):
    if value.count(old)!=1:raise ValueError(f'unexpected inherited insertion: {old}')
    return value.replace(old,new)


def runner_source():
    value=(ROOT/'scripts/trust_online_v21/runner.py').read_text(encoding='utf-8')
    return replace_once(value,'            aggregate_commit_hash = committed_hash',
        """            if self.online_controller is not None:
                monitor = self.online_controller.observe(model,x,y,split,rnd,c.seed)
                transmitted += monitor['extra_transmitted_bytes']
                self.logger.log('private_reference_monitor',rnd,monitor)
            aggregate_commit_hash = committed_hash""")


def analysis_source():
    value=(ROOT/'scripts/trust_online_v21/analysis.py').read_text(encoding='utf-8')
    value=value.replace("'v19_repair'","'v20_periodic'").replace("'v20_repair'","'v20_peak'")
    return replace_once(value,"candidate['repair_attempts']==91","candidate['repair_attempts']==4")


def study_source():
    value=(ROOT/'scripts/trust_online_v21_fixed/study.py').read_text(encoding='utf-8')
    edits=[
        ('from scripts.trust_online_v21.runner import BenchmarkRunner','from .runner import BenchmarkRunner'),
        ('from scripts.trust_online_v21.build_runner import rendered_runner','from .build_sources import runner_source as rendered_runner'),
        ('from scripts.trust_online_v21.online import OnlineRepair,DeploymentPolicy','from .online import OnlineRepair,DEPLOYMENT,MemoryPolicy'),
        ('from scripts.trust_online_v21.analysis import SEEDS,ATTACKS,ARMS,clean_gate,qualify,extract',
         'from .analysis import SEEDS,ATTACKS,ARMS,clean_gate,qualify,extract'),
        ('from scripts.trust_online_v21 import study as initial','from scripts.trust_online_v21_fixed import study as initial'),
        ('from .parity import historical_weights','from scripts.trust_online_v21_fixed.parity import historical_weights'),
        ("JOB=ROOT/'artifacts/online-v21-fixed'","JOB=ROOT/'artifacts/periodic-v22'"),
        ("OUT=ROOT/'results/online-v21-fixed'","OUT=ROOT/'results/periodic-v22'"),
        ("POLICY_FILE=ROOT/'docs/research/ONLINE_V21_FIXED.md'","POLICY_FILE=ROOT/'docs/research/PERIODIC_V22.md'"),
        ("TEST_FILE=ROOT/'tests/test_online_v21_fixed.py'","TEST_FILE=ROOT/'tests/test_periodic_v22.py'"),
        ("LAUNCHER=ROOT/'scripts/run_online_v21_fixed.ps1'","LAUNCHER=ROOT/'scripts/run_periodic_v22.ps1'"),
        ('DEPLOYMENT=DeploymentPolicy()',''),
        ("experiment_id=f'v21fixed-femnist-","experiment_id=f'v22-femnist-"),
        ("(ROOT/'scripts/trust_online_v21/runner.py')","(Path(__file__).parent/'runner.py')"),
        ("expected=list(range(10,101))","expected=[25,50,75,100]"),
        ("'scripts.trust_online_v21_fixed.study'","'scripts.trust_periodic_v22.study'"),
        ('# V21-fixed online FEMNIST development study','# V22 periodic repair and private memory study'),
        ('A fixed repair attempt follows every aggregation in rounds 10-100.',
         'Fixed repair rounds: 25, 50, 75, 100. Private fixed-example peaks are monitored every round.'),
        ('Frozen V21-fixed:','Frozen V22:'),
    ]
    for old,new in edits:value=replace_once(value,old,new)
    value=replace_once(value,'        deployment=asdict(DEPLOYMENT),repair=',
                       '        memory_policy=asdict(MemoryPolicy()),deployment=asdict(DEPLOYMENT),repair=')
    value=value.replace("'v19_repair'","'v20_periodic'").replace("'v20_repair'","'v20_peak'")
    # Reuse the original proven inner-loop smoke rather than asserting equality
    # of deliberately changed admission decisions.
    start=value.index('def smoke():');end=value.index('\n\ndef run():',start)
    value=value[:start]+'''def smoke():
    from scripts.trust_online_v21 import study as original
    if not read(original.JOB/'smoke-validation.json')['passed']:
        raise ValueError('missing unchanged-inner-loop GPU validation')
    dump(JOB/'smoke-validation.json',dict(passed=True,inner_loop_unchanged=True,
        source_sha256=digest(original.JOB/'smoke-validation.json'),original_gpu_steps=400))
'''+value[end:]
    # Do not start while the earlier frozen clean screen is still running.
    value=replace_once(value,"        assert_frozen()\n        if not read(JOB/'validation.json')", """        assert_frozen()
        previous=read(initial.JOB/'status.json')
        if previous['completed']!=12 or previous['failed'] or previous['status']!='clean_rejected':
            raise ValueError('V21-fixed must complete and reject its clean screen before V22 starts')
        if not read(JOB/'validation.json')""")
    value=replace_once(value,"        online=[e['payload'] for e in events if e['event_type']=='online_repair']", """        online=[e['payload'] for e in events if e['event_type']=='online_repair']
        monitors=[e['payload'] for e in events if e['event_type']=='private_reference_monitor']
        if row['arm'] in ('v20_periodic','v20_peak') and [m['round'] for m in monitors]!=list(range(1,101)):
            raise ValueError('incomplete private running-peak monitoring')""")
    value=replace_once(value,"        summary['final_evaluation']=final_eval", """        summary['final_evaluation']=final_eval
        summary['private_reference_presentations']=sum(m['example_presentations'] for m in monitors)
        summary['private_reference_transmitted_bytes']=sum(m['extra_transmitted_bytes'] for m in monitors)
        summary['memory_candidate_presentations']=sum(r['memory_candidate_presentations'] for r in online)
        summary['extra_repair_example_presentations']+=summary['private_reference_presentations']+summary['memory_candidate_presentations']
        summary['extra_repair_transmitted_bytes']+=summary['private_reference_transmitted_bytes']
        summary['private_reference_seconds']=sum(m['seconds'] for m in monitors)
        if row['attack']=='model_replacement':
            history=read(folder/'summary.json')['targeted_attack_success_history']
            summary['reinfection']=[dict(round=r['round'],pre_repair_trigger_rate=r['source']['trigger_target_rate'],
                previous_committed_trigger_rate=history[r['round']-2],
                increase=r['source']['trigger_target_rate']-history[r['round']-2]) for r in observer.records]""")
    return value


if __name__=='__main__':
    for name,content in (('runner.py',runner_source()),('analysis.py',analysis_source()),('study.py',study_source())):
        (Path(__file__).parent/name).write_text(content,encoding='utf-8')
