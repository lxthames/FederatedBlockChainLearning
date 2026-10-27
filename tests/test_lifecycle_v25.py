from dataclasses import replace
import json
import numpy as np
import pytest
import torch
from scripts.trust_lifecycle_v25.study import Runner,config,manifest
from scripts.trust_lifecycle_v25.baselines import RepairController,prune_units
from scripts.trust_online_v21.online import DeploymentPolicy
from scripts.trust_supervised_v18.repair import RepairPolicy
from scripts.trust_repair_v16_fixed.research.models import create_model
from scripts.trust_lifecycle_v25.data import DATA,SEEDS


@pytest.mark.parametrize('arm',['quorum','clipping','periodic','forced_periodic','finetune','forced_finetune','fineprune','forced_fineprune'])
def test_full_runner_short_trajectory_and_repair_integrity(tmp_path,arm):
    torch.set_num_threads(2)
    row=dict(dataset='fashion_mnist',seed=991,condition='attack_stop',trigger='checkerboard',arm=arm,id='smoke-'+arm)
    c=replace(config(row),device='cpu',rounds=2,num_clients=20,participation_rate=.2,
        dirichlet_alpha=None,model='default',attack_start_round=1,batch_size=16,optimizer_momentum=0.)
    rng=np.random.default_rng(991)
    x=rng.random((1000,1,28,28),dtype=np.float32);y=np.tile(np.arange(10),100)
    tx=rng.random((100,1,28,28),dtype=np.float32);ty=np.tile(np.arange(10),10)
    r=Runner(c,tmp_path/arm);r.trigger='checkerboard';r.attack_stop_round=1
    r.challenge_arm='clip_only' if arm=='clipping' else 'current'
    if arm not in ('clipping','quorum'):
        r.online_controller=RepairController(arm.removeprefix('forced_'),arm.startswith('forced_'),
            RepairPolicy(repair_clients=2,panel_size=2,examples_per_client=4,passes=1,steps_per_client=1,batch_size=4,proposals=1),
            DeploymentPolicy(first_round=1,interval=1,final_round=2))
    result=r.run_arrays(x,y,lambda:create_model('fashion_mnist',(1,28,28),10),(tx,ty))
    assert result['rounds_completed']==2
    events=[json.loads(line) for line in (tmp_path/arm/'events.jsonl').read_text().splitlines()]
    assert sum(e['event_type']=='model_integrity' and e['payload']['passed'] for e in events)==2
    repairs=[e['payload'] for e in events if e['event_type']=='online_repair']
    assert len(repairs)==(0 if arm in ('clipping','quorum') else 2)
    if arm.startswith('forced_'): assert all(e['decision']['passed'] for e in repairs)


def test_pruned_units_remain_dead_after_sgd():
    model=create_model('fashion_mnist',(1,28,28),10)
    prune_units(model,[0,2]);optimizer=torch.optim.SGD(model.parameters(),lr=.1,momentum=.9)
    for _ in range(2):
        optimizer.zero_grad();model(torch.rand(4,1,28,28)).sum().backward();optimizer.step()
    assert torch.count_nonzero(model[7].weight[[0,2]])==0
    assert torch.count_nonzero(model[9].weight[:,[0,2]])==0


def test_fresh_writer_cohorts_and_manifest():
    p=json.loads((DATA/'provenance.json').read_text())
    used=set(p['exposure_audit']['excluded_writer_ids'])
    assert not set(SEEDS).intersection(p['exposure_audit']['previous_seeds'])
    for row in p['seeds']:
        writers=set(row['writers'])|set(row['reference_writers'])
        assert not writers.intersection(used)
        used.update(writers)
    assert len(manifest())==400
    assert len({row['id'] for row in manifest()})==400
