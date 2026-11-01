"""Independent numerical audit of submitted MDS outputs against SciPy distances."""
from pathlib import Path
import json
import numpy as np
from scipy.spatial.distance import cdist
ROOT=Path(__file__).resolve().parents[1]

def main():
    metrics=json.loads((ROOT/'results/stage4/metrics.json').read_text())
    checks=[]
    for slug in ['wine','seeds','glass']:
        x=np.loadtxt(ROOT/f'data/stage4/{slug}.csv',delimiter=',',skiprows=1)
        for q in ['1','1.5','2','3','4','inf']:
            d=cdist(x,x,metric='chebyshev') if q=='inf' else cdist(x,x,metric='minkowski',p=float(q))
            sq=d*d;n=len(x);j=np.eye(n)-np.ones((n,n))/n
            gram=-.5*j@sq@j
            spectrum=np.linalg.eigvalsh((gram+gram.T)/2)[::-1]
            record=next(v for v in metrics['metrics'] if v['dataset']==slug and v['q']==q)
            for k in [2,3]:
                stem=ROOT/f'results/stage4/{slug}_mds_q{q}_{k}D'
                z=np.loadtxt(str(stem)+'.csv',delimiter=',',skiprows=1,ndmin=2)
                diag=json.loads(Path(str(stem)+'.json').read_text())
                spectral_error=float(np.linalg.norm(np.asarray(diag['eigenvalues'])-spectrum)/np.linalg.norm(spectrum))
                distance_error=float(np.linalg.norm(cdist(z,z)-d)/np.linalg.norm(d))
                assert spectral_error<1e-12,(slug,q,k,spectral_error)
                assert abs(distance_error-record[f'error_{k}D'])<1e-12
                np.testing.assert_allclose(z.mean(0),0,atol=1e-10)
                assert z.shape==(n,k) and np.isfinite(z).all()
                checks.append(dict(dataset=slug,q=q,dimensions=k,relative_spectrum_discrepancy=spectral_error,independent_normalized_distance_error=distance_error))
    (ROOT/'results/independent_audit.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(f'Independent audit passed for {len(checks)} embeddings; maximum relative spectrum discrepancy '+str(max(c['relative_spectrum_discrepancy'] for c in checks)))
if __name__=='__main__':main()
