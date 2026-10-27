import json
from pathlib import Path
from datetime import datetime, timezone
root=Path(__file__).resolve().parents[1]
status={
 'status':'blocked_missing_baseline_adapters',
 'updated_utc':datetime.now(timezone.utc).isoformat(),
 'requested_matrix':{
   'datasets':['femnist','fashion_mnist'],
   'conditions':['clean','attack_stop','continuous'],
   'triggers':['patch','checkerboard'],
   'seeds':5,
   'methods':['quorum','periodic','NAD','FLIP_or_BaFFLe']
 },
 'available_v25_methods':['quorum','clipping','periodic','clean_finetune','fine_pruning'],
 'missing_methods':['NAD','FLIP','BaFFLe'],
 'completed_v25':'400/400 trajectories; 400/400 integrity checks',
 'reason':'No faithful NAD, FLIP, or BaFFLe implementation is present in the lifecycle runner. Existing V25 arms cannot be relabeled as those baselines.',
 'runs_started':0,
 'next_valid_action':'Implement and audit a faithful NAD adapter and one documented FLIP or BaFFLe adapter, then freeze a new manifest before execution.'
}
(root/'artifacts/submission-baselines/status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
print(json.dumps(status,indent=2))
