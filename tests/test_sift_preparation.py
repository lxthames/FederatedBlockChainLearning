import io
import tarfile
import numpy as np
import pytest
from scripts.data_preparation.sift import vectors,verify_exact,target_counts,extract_vectors,NAMES


def test_fvecs_reader_preserves_canonical_row_order(tmp_path):
    expected=np.array([[1.,2.],[3.,4.]],dtype='<f4')
    records=np.empty((2,3),dtype='<i4')
    records[:,0]=2; records[:,1:]=expected.view('<i4')
    path=tmp_path/'vectors.fvecs'; records.tofile(path)
    assert np.array_equal(vectors(path),expected)
    records[1,0]=3; records.tofile(path)
    with pytest.raises(ValueError,match='dimensions'): vectors(path)


def test_exactness_check_accepts_boundary_ties_and_rejects_wrong_neighbors():
    base=np.array([[0.,0.],[1.,0.],[-1.,0.],[10.,0.]],dtype=np.float32)
    verify_exact(base,np.array([0.,0.]),np.array([0,2]))
    with pytest.raises(ValueError,match='disagrees'):
        verify_exact(base,np.array([0.,0.]),np.array([0,3]))
    with pytest.raises(ValueError,match='duplicate'):
        verify_exact(base,np.array([0.,0.]),np.array([0,0]))


def test_targets_use_actual_neighbor_ids_and_shard_assignments():
    neighbors=np.array([[0,1,4],[2,3,4]])
    assignments=np.array([0,0,1,1,2])
    assert np.array_equal(target_counts(neighbors,assignments,3),[[2,0,1],[0,2,1]])


def test_archive_ignores_arbitrary_paths_and_extracts_only_fixed_names(tmp_path):
    archive=tmp_path/'source.tar.gz'
    with tarfile.open(archive,'w:gz') as out:
        for name in (*NAMES,'../../unexpected.txt'):
            info=tarfile.TarInfo('sift/'+name); info.size=4
            out.addfile(info,io.BytesIO(b'abcd'))
    destination=tmp_path/'raw'; extract_vectors(archive,destination)
    assert {p.name for p in destination.iterdir()}==set(NAMES)
    assert not (tmp_path/'unexpected.txt').exists()
