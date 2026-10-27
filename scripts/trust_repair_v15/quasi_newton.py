"""Matrix-free Hessian products; this is NOT an inverse-Hessian optimizer."""
from collections import deque
import numpy as np


class SecantHessian:
    def __init__(self, memory=5):
        self.pairs=deque(maxlen=memory)
        self.rejected=0

    def add(self,s,y):
        s=np.asarray(s,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
        if s.ndim!=1 or s.shape!=y.shape or not np.isfinite(s).all() or not np.isfinite(y).all():
            raise ValueError('finite paired secant vectors required')
        # Nonconvex CNNs need safeguards; skipped pairs are counted, not hidden.
        if float(s@y)<=1e-10*max(float(np.linalg.norm(s)*np.linalg.norm(y)),1e-12):
            self.rejected+=1;return False
        self.pairs.append((s.copy(),y.copy()));return True

    def product(self,v):
        if not self.pairs:raise ValueError('no positive-curvature secants')
        v=np.asarray(v,dtype=np.float64)
        s,y=self.pairs[-1];gamma=float(s@y/(s@s))
        terms=[]
        def apply(q):
            answer=gamma*q.copy()
            for bs,sbs,yy,ys in terms:
                answer-=bs*float(bs@q)/sbs
                answer+=yy*float(yy@q)/ys
            return answer
        for s,y in self.pairs:
            bs=apply(s);sbs=float(s@bs);ys=float(s@y)
            if sbs<=0 or not np.isfinite(sbs):raise ValueError('invalid Hessian curvature')
            terms.append((bs,sbs,y,ys))
        answer=apply(v)
        if not np.isfinite(answer).all():raise ValueError('nonfinite Hessian product')
        return answer
