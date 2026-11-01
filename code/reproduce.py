"""Reproduce all required results offline from the submitted UCI matrices."""
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('MPLCONFIGDIR','/tmp/project1-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/project1-cache')

def run(args,cwd=ROOT):
    p=subprocess.run([sys.executable,*args],cwd=cwd,capture_output=True,text=True)
    if p.returncode:
        print(p.stdout);print(p.stderr,file=sys.stderr);raise SystemExit(p.returncode)
    return p.stdout+p.stderr

def main():
    out=ROOT/'results';out.mkdir(exist_ok=True)
    test=run(['code/test_mds.py']);(out/'test_results.txt').write_text(test)
    print(test,flush=True)
    pca=run(['code/pca.py','--input','data/Iris.csv','--output-dir','results'])
    (out/'Iris_pca_stdout.json').write_text(pca)
    for k in [2,3]:
        log=run(['../code/mds.py','--input','../data/Iris.csv','--q','2','--dimensions',str(k)],out)
        (out/f'Iris_mds_q2_{k}D.json').write_text(log)
    print(run(['code/investigate.py']),flush=True)
    run(['code/comparison_panels.py'])
    print('All result generation completed.',flush=True)
if __name__=='__main__':main()
