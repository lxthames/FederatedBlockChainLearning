import hashlib
import json

import pytest
import torch

from scripts.research.attacks import ALIEAttack, AttackContext
from scripts.research.contracts import ClientUpdate
from scripts.research import workflow
from scripts.research.paper_artifacts import paired_statistics
from scripts.research.trust_matrix import SEEDS, cell


@pytest.mark.parametrize('peer_count',[0,1])
def test_alie_shortfall_is_explicit_and_does_not_fabricate_poison(peer_count):
    update=ClientUpdate(0,{'w':torch.tensor([2.,3.])},5,True)
    peers=tuple(ClientUpdate(i+1,{'w':torch.ones(2)},5) for i in range(peer_count))
    result=ALIEAttack().apply(update,AttackContext(1,peer_updates=peers))
    assert torch.equal(result.state['w'],update.state['w'])
    assert result.metadata['attack_applied'] is False
    assert result.metadata['reference_peer_count']==peer_count


def test_resume_rejects_legacy_source_even_when_config_matches(tmp_path,monkeypatch):
    config=cell('test')
    monkeypatch.setattr(workflow,'RESULTS',tmp_path)
    monkeypatch.setattr(workflow,'source_hash',lambda:'current')
    folder=tmp_path/'core_confirmation'/config.experiment_id
    folder.mkdir(parents=True)
    (folder/'summary.json').write_text(json.dumps(dict(
        config_hash=hashlib.sha256(json.dumps(config.to_dict(),sort_keys=True).encode()).hexdigest(),
        source_hash='a13aeacc130f6a7cb780e9ccbab0e47ab08bc1b9aeda46a127c33bcfb1ef3b1d',
        status='completed',rounds_completed=config.rounds)))
    with pytest.raises(ValueError,match='different research code'):
        workflow.execute(config,'core_confirmation')


def test_paired_analysis_never_combines_different_source_versions():
    summaries=[]
    for method,source in [('two_layer','new'),('fedavg','old')]:
        for seed in SEEDS:
            summaries.append(dict(configuration={'defense':method,'seed':seed},
                defense=method,seed=seed,source_hash=source,final_utility=.5,
                data_fingerprint='data',test_fingerprint='test'))
    assert paired_statistics(summaries)==[]
