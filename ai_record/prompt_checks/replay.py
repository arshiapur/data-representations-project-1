"""Replay this supplementary non-Iris portability check from inspected UCI sources.
This is a dataset-specific audit replay, not a replacement for Prompt1's adaptive acquisition.
"""
import csv,json,os,subprocess,sys,zipfile
from pathlib import Path
import numpy as np
r=Path(__file__).resolve().parent
repo=r.parents[1]
query='wisconsin_cancer_diagnostic'
base=['radius','texture','perimeter','area','smoothness','compactness','concavity','concave_points','symmetry','fractal_dimension']
names=[f'{f}_{summary}' for summary in ['mean','se','worst'] for f in base]
with zipfile.ZipFile(r/'source.zip') as z: rows=list(csv.reader(z.read('wdbc.data').decode().splitlines()))
x=np.asarray([[float(v) for v in row[2:]] for row in rows])
assert x.shape==(569,30) and np.isfinite(x).all()
with (r/'data'/f'{query}.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(names);w.writerows(x)
for i in [1,2]:(r/f'Prompt{i}_executed.txt').write_text((repo/f'prompts/Prompt{i}.txt').read_text()+'\n'+query+'\n')
env=os.environ.copy();env.update(MPLCONFIGDIR='/tmp/project1-matplotlib',XDG_CACHE_HOME='/tmp/project1-cache',OPENBLAS_NUM_THREADS='2')
p=subprocess.run([sys.executable,str(repo/'code/pca.py'),'--input',str(r/'data'/f'{query}.csv'),'--output-dir',str(r/'results')],env=env,capture_output=True,text=True)
(r/'pca_stdout.txt').write_text(p.stdout+p.stderr)
if p.returncode:raise SystemExit(p.returncode)
print(p.stdout)
