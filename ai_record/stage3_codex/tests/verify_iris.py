"""Independent centered SVD validation, orthogonal alignment without scaling."""
import json
from pathlib import Path
import numpy as np
root = Path(__file__).resolve().parents[1]
x = np.loadtxt(root/'data/Iris.csv',delimiter=',',skiprows=1)
xc = x-x.mean(axis=0)
u,s,vt = np.linalg.svd(xc, full_matrices=False)
results = {}
for k in (2,3):
    y = np.loadtxt(root/f'results/Iris_mds_q2_{k}D.csv',delimiter=',',skiprows=1)
    reference = u[:,:k]*s[:k]
    left,_,right = np.linalg.svd(y.T@reference)
    aligned = y@(left@right)
    error = np.linalg.norm(aligned-reference)/np.linalg.norm(reference)
    assert error < 1e-12, error
    results[str(k)] = {'orthogonal_alignment_relative_frobenius_error':float(error),'scaling_applied':False,'max_absolute_coordinate_error':float(np.max(np.abs(aligned-reference)))}
print(json.dumps(results,indent=2))
