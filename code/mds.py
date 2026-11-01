"""Classical MDS / STRAIN, following Multidimensional-Scaling.pdf pp. 2–4.
Rows are observations (transpose of the handout's column-vector convention).
No high-level MDS/PCA/classical-scaling routine is used.
"""
import argparse
import csv
from decimal import Decimal, InvalidOperation
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/project1-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/project1-cache')
import numpy as np
from scipy.linalg import eigh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_q(value):
    if value == 'inf':
        return np.inf, 'inf'
    try:
        d = Decimal(value)
        if not d.is_finite() or d < 1:
            raise ValueError('q must be finite and >=1, or literal inf')
        q = float(d)
        if not np.isfinite(q):
            raise ValueError('q is outside supported floating-point range')
        tag = format(d, 'f')
        if '.' in tag:
            tag = tag.rstrip('0').rstrip('.')
        return q, tag
    except (InvalidOperation, OverflowError) as e:
        raise ValueError('q must be a real >=1, or literal inf') from e


def read_matrix(path):
    with Path(path).open(newline='') as f:
        rows=csv.reader(f)
        names=next(rows, [])
        values=list(rows)
    if not names or len(set(names)) != len(names) or any(not s.strip() for s in names):
        raise ValueError('CSV needs unique nonempty feature headers')
    if not values or any(len(row)!=len(names) for row in values):
        raise ValueError('CSV is empty or has inconsistent row lengths')
    x=np.asarray(values,dtype=float)
    if len(x)<2 or not np.isfinite(x).all():
        raise ValueError('Need >=2 observations with finite numeric features')
    return x


def distances(x,q):
    x=np.asarray(x,dtype=float)
    if x.ndim!=2 or x.shape[1]<1 or not np.isfinite(x).all():
        raise ValueError('Expected finite observation-feature matrix')
    if not (q>=1) or np.isnan(q):
        raise ValueError('q must be >=1')
    with np.errstate(over='raise',invalid='raise'):
        delta=np.abs(x[:,None,:]-x[None,:,:])
        scale=delta.max(axis=2)
        if np.isinf(q):
            return scale
        # Factoring out the largest coordinate prevents overflow in delta**q.
        ratio=np.divide(delta,scale[:,:,None],out=np.zeros_like(delta),where=scale[:,:,None]!=0)
        result=scale*np.sum(ratio**q,axis=2)**(1/q)
    np.fill_diagonal(result,0.)
    return result


def classical_mds(d,k):
    d=np.asarray(d,dtype=float)
    if k not in (2,3): raise ValueError('dimensions must be 2 or 3')
    if d.ndim!=2 or d.shape[0]!=d.shape[1] or not np.isfinite(d).all() or (d<0).any():
        raise ValueError('Distance matrix must be finite, square and nonnegative')
    if not np.allclose(d,d.T,rtol=1e-12,atol=1e-12) or not np.allclose(np.diag(d),0,atol=1e-12):
        raise ValueError('Distances must be symmetric with zero diagonal')
    a=d*d
    # Handout eq. (6): -aij/2 + ai/(2n) + aj/(2n) - a/(2n²).
    g=-.5*(a-a.mean(axis=1,keepdims=True)-a.mean(axis=0,keepdims=True)+a.mean())
    g=(g+g.T)/2
    eigenvalues,vectors=eigh(g,check_finite=True)
    eigenvalues=eigenvalues[::-1]
    vectors=vectors[:,::-1]
    # 100*n*eps times spectral scale: conservative roundoff threshold.
    tol=100*len(d)*np.finfo(float).eps*max(float(np.max(np.abs(eigenvalues))),np.finfo(float).tiny)
    meaningful=eigenvalues>tol
    # Handout clipping: negative eigenvalues become zero. Positive roundoff
    # values are retained for coordinates; tolerance only classifies diagnostics.
    take=min(k,len(d))
    z=np.zeros((len(d),k))
    z[:,:take]=vectors[:,:take]*np.sqrt(np.maximum(eigenvalues[:take],0))
    positive_mass=float(eigenvalues[meaningful].sum())
    negative=eigenvalues < -tol
    negative_mass=float(-eigenvalues[negative].sum())
    diag=dict(eigenvalues=eigenvalues.tolist(),tolerance=tol,negative_count=int(negative.sum()),negative_mass=negative_mass,negative_mass_fraction=negative_mass/(negative_mass+positive_mass) if positive_mass+negative_mass else 0.,positive_mass=positive_mass,positive_rank=int(meaningful.sum()))
    for dim in (2,3):
        mass=float(np.where(meaningful,eigenvalues,0)[:dim].sum())
        diag[f'positive_mass_fraction_{dim}D']=mass/positive_mass if positive_mass else None
    den=float(np.linalg.norm(d))
    diag['normalized_distance_error']=float(np.linalg.norm(distances(z,2)-d)/den) if den else 0.
    return z,diag


def plot(z,path,title):
    k=z.shape[1]
    fig=plt.figure(figsize=(7,7))
    ax=fig.add_subplot(111,projection='3d' if k==3 else None)
    ax.scatter(*z.T,s=15,color='steelblue',alpha=.8)
    mid=(z.min(0)+z.max(0))/2
    half=max(float(np.ptp(z,axis=0).max())*.55,1e-12)
    setters=[ax.set_xlim,ax.set_ylim]+([ax.set_zlim] if k==3 else [])
    labels=[ax.set_xlabel,ax.set_ylabel]+([ax.set_zlabel] if k==3 else [])
    for i,(setter,label) in enumerate(zip(setters,labels)):
        setter(mid[i]-half,mid[i]+half);label(f'MDS{i+1}')
    if k==3: ax.set_box_aspect((1,1,1))
    else: ax.set_aspect('equal',adjustable='box')
    ax.set_title(title);fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--q',required=True)
    p.add_argument('--dimensions',type=int,choices=(2,3),required=True)
    args=p.parse_args()
    try:
        q,tag=parse_q(args.q)
        if args.input.suffix!='.csv': raise ValueError('Input must end in .csv')
        x=read_matrix(args.input)
        d=distances(x,q)
        z,diag=classical_mds(d,args.dimensions)
        stem=f'{args.input.stem}_mds_q{tag}_{args.dimensions}D'
        with open(stem+'.csv','w',newline='') as f:
            w=csv.writer(f);w.writerow([f'MDS{i+1}' for i in range(args.dimensions)]);w.writerows(z)
        plot(z,stem+'.png',f'{args.input.stem}: classical MDS, q={tag} ({args.dimensions}D)')
        diag.update(input=str(args.input),q=tag,dimensions=args.dimensions,coordinates=stem+'.csv',image=stem+'.png')
        print(json.dumps(diag,allow_nan=False))
    except (ValueError,OSError,FloatingPointError) as e:
        p.error(str(e))
if __name__=='__main__': main()
