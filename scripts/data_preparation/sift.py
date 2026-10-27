"""Download official SIFT1M and rebuild exact, reproducible routing targets.

This tool writes a separate dataset root. It never certifies or overwrites the
old project-specific shard labels. Exact search runs on CPU, leaving CUDA free.
"""
import argparse
import datetime
import ftplib
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import time
import traceback
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
JOB=ROOT/'artifacts/sift-data-preparation'
DATA=ROOT/'data/sift1m-verified'
SOURCE='ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz'
SOURCE_PATH='/local/texmex/corpus/sift.tar.gz'
NAMES=('sift_base.fvecs','sift_learn.fvecs','sift_query.fvecs','sift_groundtruth.ivecs')


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda:source.read(4*1024*1024),b''): digest.update(block)
    return digest.hexdigest()


def atomic_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,sort_keys=True),encoding='utf-8')
    temporary.replace(path)


def vectors(path, floating=True):
    raw=np.memmap(path,dtype='<i4',mode='r')
    if not raw.size: raise ValueError('empty vector file')
    dimensions=int(raw[0])
    if dimensions<1 or raw.size%(dimensions+1):
        raise ValueError('invalid vector record lengths')
    rows=raw.reshape(-1,dimensions+1)
    if not np.all(rows[:,0]==dimensions): raise ValueError('inconsistent vector dimensions')
    values=rows[:,1:]
    return values.view('<f4') if floating else values


def extract_vectors(archive,directory):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    found=set()
    with tarfile.open(archive,'r:gz') as source:
        for member in source:
            name=Path(member.name).name
            if name not in NAMES: continue
            if not member.isfile() or name in found:
                raise ValueError('unexpected or duplicate archive entry')
            # Copy only whitelisted regular files to fixed destinations; never
            # use archive-provided paths, symlinks, or extractall.
            with source.extractfile(member) as inp, (directory/name).open('wb') as out:
                shutil.copyfileobj(inp,out,4*1024*1024)
            if (directory/name).stat().st_size!=member.size:
                raise ValueError('incomplete extracted vector file')
            found.add(name)
    if found!=set(NAMES): raise ValueError('official archive is missing expected files')


def download(archive,update):
    archive=Path(archive); archive.parent.mkdir(parents=True,exist_ok=True)
    partial=archive.with_suffix(archive.suffix+'.part')
    for attempt in range(5):
        try:
            with ftplib.FTP('ftp.irisa.fr',timeout=30) as ftp:
                ftp.login(); ftp.voidcmd('TYPE I')
                total=ftp.size(SOURCE_PATH)
                remote_mdtm=ftp.sendcmd('MDTM '+SOURCE_PATH)
                identity=archive.parent/'remote-identity.json'
                metadata={'source_url':SOURCE,'size':total,'mdtm':remote_mdtm}
                if identity.exists() and json.loads(identity.read_text())!=metadata:
                    raise ValueError('remote archive changed; retain old partial and start a new dataset version')
                atomic_json(identity,metadata)
                if archive.exists():
                    if archive.stat().st_size!=total: raise ValueError('existing archive has wrong size')
                    update(download_bytes=total,download_total_bytes=total)
                    return
                offset=partial.stat().st_size if partial.exists() else 0
                if offset>total: raise ValueError('partial archive exceeds remote size')
                update(download_bytes=offset,download_total_bytes=total,source_url=SOURCE)
                last_update=time.monotonic()
                with partial.open('ab') as out:
                    def receive(block):
                        nonlocal offset,last_update
                        out.write(block); offset+=len(block)
                        if time.monotonic()-last_update>=2:
                            out.flush(); update(download_bytes=offset)
                            last_update=time.monotonic()
                    if offset<total:
                        ftp.retrbinary('RETR '+SOURCE_PATH,receive,blocksize=256*1024,rest=offset or None)
                if partial.stat().st_size!=total: raise ValueError('incomplete archive download')
                partial.replace(archive)
                update(download_bytes=total,download_total_bytes=total)
                return
        except ValueError:
            raise
        except Exception as error:
            update(download_attempt=attempt+1,last_network_error=str(error))
            if attempt==4: raise
            time.sleep(2*(attempt+1))


def exact_neighbors(index,queries,path,k,update,phase):
    path=Path(path)
    progress=path.with_suffix('.progress.json')
    completed=0
    if path.exists() and progress.exists():
        saved=json.loads(progress.read_text())
        if saved['shape']!=[len(queries),k]: raise ValueError('neighbor resume shape mismatch')
        completed=saved['completed']
        result=np.lib.format.open_memmap(path,mode='r+')
    else:
        result=np.lib.format.open_memmap(path,mode='w+',dtype=np.int32,shape=(len(queries),k))
    for start in range(completed,len(queries),128):
        end=min(start+128,len(queries))
        _,ids=index.search(np.ascontiguousarray(queries[start:end],dtype=np.float32),k)
        if np.any(ids<0): raise ValueError('exact search returned insufficient neighbors')
        result[start:end]=ids
        result.flush()
        atomic_json(progress,{'completed':end,'shape':[len(queries),k]})
        update(phase=phase,queries_completed=end,total_queries=len(queries))
    return result


def verify_exact(base,query,indices):
    indices=np.asarray(indices)
    if len(set(indices.tolist()))!=len(indices) or np.any(indices<0) or np.any(indices>=len(base)):
        raise ValueError('invalid or duplicate nearest-neighbor IDs')
    distances=np.empty(len(base),dtype=np.float64)
    for start in range(0,len(base),16384):
        difference=np.asarray(base[start:start+16384],dtype=np.float64)-np.asarray(query,dtype=np.float64)
        distances[start:start+len(difference)]=np.einsum('ij,ij->i',difference,difference)
    boundary=np.partition(distances,len(indices)-1)[len(indices)-1]
    if np.max(distances[indices])>boundary+1e-7:
        raise ValueError('exact-neighbor search disagrees with independent float64 distances')


def target_counts(neighbors,assignment,population):
    neighbors=np.asarray(neighbors)
    assignment=np.asarray(assignment)
    if neighbors.ndim!=2 or neighbors.min()<0 or neighbors.max()>=len(assignment):
        raise ValueError('neighbor IDs outside database assignments')
    targets=np.empty((len(neighbors),population),dtype=np.int32)
    for row,ids in enumerate(neighbors):
        targets[row]=np.bincount(assignment[ids],minlength=population)
    if not np.all(targets.sum(1)==neighbors.shape[1]): raise ValueError('incorrect target mass')
    return targets


def prepare(k=1000,threads=4):
    import faiss
    JOB.mkdir(parents=True,exist_ok=True); DATA.mkdir(parents=True,exist_ok=True)
    if (DATA/'sift-provenance.json').exists():
        raise FileExistsError('verified output already exists; use it or create a new dataset version')
    state={'status':'running','pid':os.getpid(),'phase':'download','dataset':'SIFT1M',
           'legacy_targets_verified':False,'query_target_correspondence_verified':False,
           'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    def update(**values):
        state.update(values); state['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        atomic_json(JOB/'status.json',state)
    update()
    try:
        if shutil.disk_usage(DATA).free<3*1024**3:
            raise ValueError('at least 3 GiB of free disk space is required')
        raw=DATA/'raw'; raw.mkdir(exist_ok=True)
        archive=raw/'sift.tar.gz'
        download(archive,update)
        update(phase='extract_and_check')
        extract_vectors(archive,raw)
        base=vectors(raw/'sift_base.fvecs')
        learn=vectors(raw/'sift_learn.fvecs')
        query=vectors(raw/'sift_query.fvecs')
        public_truth=vectors(raw/'sift_groundtruth.ivecs',floating=False)
        if base.shape!=(1000000,128) or query.shape!=(10000,128) or learn.shape[1]!=128:
            raise ValueError('unexpected canonical SIFT1M array shapes')
        if len(public_truth)!=len(query) or not 1<=k<=len(base):
            raise ValueError('invalid ground truth or search budget')
        if not all(np.isfinite(a).all() for a in (base,learn,query)):
            raise ValueError('non-finite source vectors')
        faiss.omp_set_num_threads(threads)
        index=faiss.IndexFlatL2(128)
        index.add(np.ascontiguousarray(base))
        train_neighbors=exact_neighbors(index,learn,DATA/'training_neighbors.npy',k,update,'exact_training_neighbors')
        test_neighbors=exact_neighbors(index,query,DATA/'test_neighbors.npy',k,update,'exact_test_neighbors')
        update(phase='independent_exactness_checks')
        for i in (0,17,123):
            verify_exact(base,query[i],test_neighbors[i])
        verify_exact(base,learn[0],train_neighbors[0])
        # Compare the public source ground truth to our independently computed
        # distances, allowing ties at its own cutoff rather than relying on IDs.
        for i in (0,17,123):
            truth=public_truth[i]
            if np.any(truth<0) or np.any(truth>=len(base)):
                raise ValueError('invalid published ground-truth index')
            verify_exact(base,query[i],truth)
        update(phase='deterministic_sharding')
        training_sample=np.random.default_rng(2027).choice(len(base),100000,replace=False)
        for population in (4,8):
            kmeans=faiss.Kmeans(128,population,niter=25,seed=2027,nredo=1,verbose=False)
            kmeans.train(np.ascontiguousarray(base[training_sample]))
            assignments=np.empty(len(base),dtype=np.int32)
            for start in range(0,len(base),32768):
                _,ids=kmeans.index.search(np.ascontiguousarray(base[start:start+32768]),1)
                assignments[start:start+len(ids)]=ids[:,0]
            np.save(DATA/f'shard_assignments_{population}.npy',assignments)
            np.save(DATA/f'shard_centroids_{population}.npy',kmeans.centroids)
            for name,neighbors in (('',train_neighbors),('test',test_neighbors)):
                targets=target_counts(neighbors,assignments,population)
                np.save(DATA/f'gt_{population}M_client_counts_skewed{name}.npy',targets)
        # Filenames retain loader compatibility only. This is explicitly SIFT1M,
        # with official learn queries as training examples, never a 1B dataset.
        np.save(DATA/'SIFT1B_train.npy',np.asarray(learn,dtype=np.float32)/255.0)
        np.save(DATA/'SIFT_test_query.npy',np.asarray(query,dtype=np.float32))
        provenance={'dataset_name':'SIFT1M','source_url':SOURCE,
            'source_archive_sha256':sha256(archive),'source_arrays':{
                name:{'sha256':sha256(raw/name)} for name in NAMES},
            'database_size':len(base),'training_query_count':len(learn),'test_query_count':len(query),
            'neighbors_per_query':k,'query_target_correspondence_verified':True,
            'exactness_method':'CPU FAISS IndexFlatL2; independent float64 exhaustive checks for sampled queries and published ground truth',
            'sharding':'4/8 centroid partitions trained only on database vectors; seed 2027; 25 iterations',
            'training_feature_transform':'official learn vectors divided by 255, saved as float32',
            'test_feature_divisor':255.0,'legacy_filenames_are_loader_aliases':True,
            'legacy_project_labels_verified':False,
            'preparation_source_sha256':sha256(__file__),
            'faiss_version':faiss.__version__,
            'generated_files':{p.name:sha256(p) for p in sorted(DATA.glob('*.npy'))}}
        atomic_json(DATA/'sift-provenance.json',provenance)
        update(status='completed',phase='ready',query_target_correspondence_verified=True,
               dataset_root=str(DATA),finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    except BaseException:
        update(status='failed',error=traceback.format_exc())
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--neighbors',type=int,default=1000)
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args()
    prepare(args.neighbors,args.threads)

if __name__=='__main__': main()
