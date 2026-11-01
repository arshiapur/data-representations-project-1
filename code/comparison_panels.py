"""Supporting 2D panels, aligned by rotation/reflection only to q=2."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/project1-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/project1-cache')
from pathlib import Path
import numpy as np
from scipy.linalg import orthogonal_procrustes
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]

def main():
    fig,axs=plt.subplots(3,3,figsize=(12,12))
    for row,slug in enumerate(['wine','seeds','glass']):
        base=np.loadtxt(ROOT/f'results/stage4/{slug}_mds_q2_2D.csv',delimiter=',',skiprows=1)
        configs=[]
        for q in ['1','2','inf']:
            z=np.loadtxt(ROOT/f'results/stage4/{slug}_mds_q{q}_2D.csv',delimiter=',',skiprows=1)
            rotation,_=orthogonal_procrustes(z,base)
            configs.append(z@rotation)
        both=np.concatenate(configs)
        half=max(np.abs(both).max()*1.05,1.)
        for col,(q,z) in enumerate(zip(['1','2','inf'],configs)):
            ax=axs[row,col];ax.scatter(*z.T,s=9,color='steelblue',alpha=.7)
            ax.set(xlim=(-half,half),ylim=(-half,half),xlabel='Aligned MDS1',ylabel='Aligned MDS2',title=f'{slug.title()}, q={q}')
            ax.set_aspect('equal',adjustable='box')
    fig.suptitle('Classical MDS: original units; orthogonal alignment only; common axes within each dataset')
    fig.tight_layout(rect=(0,0,1,.97))
    out=ROOT/'results/supporting_images';out.mkdir(exist_ok=True)
    fig.savefig(out/'q_comparison_2D.png',dpi=160);plt.close(fig)
if __name__=='__main__':main()
