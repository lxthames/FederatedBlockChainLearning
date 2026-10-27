import hashlib
from pathlib import Path
import pytest
from scripts.data_preparation.run_sift_study import recorded_files_match,configs,attack_gates


def test_label_hashes_detect_corruption_and_refuse_parent_paths(tmp_path):
    path=tmp_path/'targets.npy'; path.write_bytes(b'verified targets')
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    recorded_files_match(tmp_path,{'targets.npy':digest})
    path.write_bytes(b'changed targets')
    with pytest.raises(ValueError,match='checksum'): recorded_files_match(tmp_path,{'targets.npy':digest})
    with pytest.raises(ValueError,match='invalid'): recorded_files_match(tmp_path,{'../targets.npy':digest})


def test_new_dataset_study_uses_separate_seeds_and_complete_matched_matrix():
    development,full=configs('cpu')
    assert len(development)==36 and len(full)==180
    assert {c.seed for c in development}=={3,5,7}
    assert {c.seed for c in full}=={13,29,47,71,101}
    assert len({c.experiment_id for c in full})==180
    assert all(c.rounds==50 and c.device=='cpu' for c in full)
    assert {c.num_clients for c in full}=={4,8}
    assert {c.model for c in full}=={'emh_dnn','mlp'}


def test_ineffective_poisoning_is_not_treated_as_validated_robustness():
    rows=[dict(num_clients=4,model='mlp',seed=s,attack=a,final_utility=0.8)
          for s in (3,5,7) for a in ('none','sign_flip','low_and_slow')]
    gates=attack_gates(rows)
    assert gates['4:mlp:none']['passed']
    assert not gates['4:mlp:sign_flip']['passed']
    assert not gates['8:emh_dnn:none']['passed']
