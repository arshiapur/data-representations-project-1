"""Replay the locked Stage 4 experiment and write its numerical report."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from scipy.linalg import orthogonal_procrustes, svd
from scipy.stats import spearmanr
from mds import distances, read_matrix
ROOT=Path(__file__).resolve().parents[1]
QS=['1','1.5','2','3','4','inf']
SLUGS=['wine','seeds','glass']

def neighborhoods(d,k=5):
    """Include ALL exact/nearly tied observations at kth-distance boundary."""
    sets=[]
    for i,row in enumerate(d):
        candidates=np.delete(row,i)
        radius=np.partition(candidates,k-1)[k-1]
        tol=1e-12*max(1.,float(radius))
        sets.append(set(np.flatnonzero((row<=radius+tol)&(np.arange(len(row))!=i))))
    return sets

def run_cli(path,q,k,out):
    cmd=[sys.executable,str(ROOT/'code/mds.py'),'--input',str(path),'--q',q,'--dimensions',str(k)]
    p=subprocess.run(cmd,cwd=out,capture_output=True,text=True,check=True)
    diag=json.loads(p.stdout)
    (out/f'{path.stem}_mds_q{q}_{k}D.json').write_text(json.dumps(diag,indent=2)+'\n')
    return diag,np.loadtxt(out/diag['coordinates'],delimiter=',',skiprows=1,ndmin=2)

def main():
    out=ROOT/'results/stage4';out.mkdir(parents=True,exist_ok=True)
    if not (ROOT/'data/stage4/selection.md').exists():raise RuntimeError('Selection must be locked before experiment')
    records=json.loads((ROOT/'data/acquisition.json').read_text())
    summaries={};table=[];validation=[]
    for slug in SLUGS:
        path=ROOT/'data/stage4'/f'{slug}.csv'
        rec=next(d for d in records if d['slug']==slug)
        assert hashlib.sha256(path.read_bytes()).hexdigest()==rec['matrix_sha256']
        x=read_matrix(path);base=distances(x,2);nb=neighborhoods(base)
        all_stability=[];perq={}
        for q in QS:
            diag2,z2=run_cli(path,q,2,out);diag3,z3=run_cli(path,q,3,out)
            d=distances(x,float(q));sets=neighborhoods(d)
            stability=np.array([len(a&b)/len(a|b) for a,b in zip(sets,nb)])
            all_stability.append(stability)
            rank=float(spearmanr(base[np.triu_indices(len(x),1)],d[np.triu_indices(len(x),1)]).statistic)
            row=dict(dataset=slug,q=q,negative_count=diag2['negative_count'],negative_mass=diag2['negative_mass'],negative_mass_fraction=diag2['negative_mass_fraction'],positive_mass_2D=diag2['positive_mass_fraction_2D'],positive_mass_3D=diag2['positive_mass_fraction_3D'],error_2D=diag2['normalized_distance_error'],error_3D=diag3['normalized_distance_error'],mean_neighbor_jaccard=float(stability.mean()),pairwise_spearman=rank)
            table.append(row);perq[q]=row
            if q=='2':
                a=x-x.mean(0);_,_,vt=svd(a,full_matrices=False)
                for k,z in [(2,z2),(3,z3)]:
                    pc=a@vt[:k].T
                    rot,_=orthogonal_procrustes(z,pc)
                    aligned=float(np.linalg.norm(z@rot-pc)/np.linalg.norm(pc))
                    pair=float(np.linalg.norm(distances(z,2)-distances(pc,2))/np.linalg.norm(distances(pc,2)))
                    if max(aligned,pair)>1e-9:raise AssertionError('q=2 PCA validation failed')
                    validation.append(dict(dataset=slug,dimensions=k,relative_aligned_error=aligned,relative_pairwise_error=pair))
        stability=np.array(all_stability)
        # Exclude the tautological q=2 comparison from per-row stability averages.
        mean=stability[[i for i,q in enumerate(QS) if q!='2']].mean(0)
        with (out/f'{slug}_neighborhood_stability.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['row_1based',*['q'+q for q in QS],'mean_non_q2']);w.writerows([[i+1,*stability[:,i],mean[i]] for i in range(len(x))])
        # Share of total unscaled squared pairwise differences, algebraically variance shares.
        variance=x.var(0);share=variance/variance.sum();order=np.argsort(-share)
        feature_rows=[dict(feature=rec['features'][i],sd=float(x.std(0)[i]),range=float(np.ptp(x,axis=0)[i]),squared_difference_share=float(share[i]),zero_fraction=float((x[:,i]==0).mean())) for i in order]
        # Exact tied extrema are all reported, avoiding an arbitrary single-row choice.
        most=(np.flatnonzero(np.isclose(mean,mean.max(),rtol=0,atol=1e-12))+1).tolist()
        least=(np.flatnonzero(np.isclose(mean,mean.min(),rtol=0,atol=1e-12))+1).tolist()
        summaries[slug]=dict(features=feature_rows,most_stable_rows=most,least_stable_rows=least,maximum_stability=float(mean.max()),minimum_stability=float(mean.min()),perq=perq)
        print(f'Completed {slug}',flush=True)
    (out/'metrics.json').write_text(json.dumps(dict(metrics=table,pca_validation=validation,summaries=summaries),indent=2)+'\n')
    with (out/'metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    write_report(table,validation,summaries)

def write_report(table,validation,summaries):
    lines=['# Minkowski geometry: Wine, Seeds and Glass Identification','',
    '## Selection and reproducibility','',
    'Exactly these three UCI datasets were locked before any q-dependent computation; see [selection record](../data/stage4/selection.md) and [source/cleaning audit](../data/acquisition.json). No replacements or dataset-specific q additions were made. Wine is 178×13, Seeds 210×7, Glass Identification 214×9. All use numeric unsupervised features in original units and source row order, with targets/IDs excluded. No missing rows, imputations or scaling. Duplicate observations are retained (one in Glass; none in Wine or Seeds). A largest 214×214 double matrix is 0.35 MiB, so explicit distances and dense eigendecomposition are feasible.','',
    '| Dataset | UCI ID / official page | Exact archive member | Selection rationale |',
    '|---|---|---|---|',
    '| Wine | [109](https://archive.ics.uci.edu/dataset/109/wine) | wine.data | 13 chemical measures with heterogeneous magnitudes |',
    '| Seeds | [236](https://archive.ics.uci.edu/dataset/236/seeds) | seeds_dataset.txt | Seven geometric features including derived compactness |',
    '| Glass Identification | [42](https://archive.ics.uci.edu/dataset/42/glass+identification) | glass.data | Oxide composition, zeros and refractive index |','',
    'Commands: `python code/investigate.py` reruns both dimensions via the exact mds.py CLI for every dataset/q and rebuilds this report. Full spectra, embedding coordinates and PNGs are retained under results/stage4/. Metrics and per-observation neighborhoods are also retained. Source archives and pages are included; no live retrieval is needed to replay submitted matrices.','',
    '## Definitions and numerical policy','',
    'We compute D_q from original feature differences (never from PCA coordinates). Classical MDS uses G = −½ J(D_q squared elementwise)J, symmetrizes G and uses scipy.linalg.eigh. The top K eigenvectors are multiplied by square roots of their eigenvalues, clipping negative values to zero, as in the handout. Small positive values are retained in coordinates; diagnostics classify eigenvalues using τ = 100 n ε max|λ| (with the smallest positive float as a zero-spectrum floor). A meaningful negative is λ < −τ. This threshold scales with the spectrum and conservatively accounts for roundoff, not scientific uncertainty.','',
    'Negative mass = Σ(−λ) over meaningful negatives. Negative fraction = negative mass/(negative mass + meaningful positive mass), which allows scale-normalized comparisons across q. Positive spectral fractions sum the leading two/three meaningful positive eigenvalues divided by total meaningful positive mass. These fractions do not measure original-distance fit when negative spectrum exists. Distance error is ||D_embedding − D_q||_F / ||D_q||_F; embedding distances are Euclidean. Classical MDS minimizes Gram/STRAIN approximation, not this distance error, so monotone improvement in distance error is not guaranteed by its objective.','',
    'Neighborhoods contain the five nearest observations, including every tie within 10^-12 max(1, fifth-distance) of the boundary; self is excluded. Agreement is set Jaccard relative to q=2, averaged over rows. Duplicate observations remain valid neighbors. Spearman uses upper-triangle pairwise distances and average ranks for ties. Stable/unstable row summaries average Jaccard over the five non-q=2 settings and use one-based final-matrix row indices; every tied extremum is listed.','',
    '## Full numerical comparison','',
    '| Dataset | q | Negative count | Negative mass | Negative fraction | Positive 2D | Positive 3D | Error 2D | Error 3D | Neighbor Jaccard | Distance rank ρ |',
    '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in table:
        lines.append(f"| {r['dataset']} | {r['q']} | {r['negative_count']} | {r['negative_mass']:.6g} | {r['negative_mass_fraction']:.5f} | {r['positive_mass_2D']:.5f} | {r['positive_mass_3D']:.5f} | {r['error_2D']:.5f} | {r['error_3D']:.5f} | {r['mean_neighbor_jaccard']:.5f} | {r['pairwise_spearman']:.5f} |")
    lines+=['','## Independent centered-PCA validation at q=2','',
    'For each unchanged X, scipy.linalg.svd(X−mean(X), full_matrices=False) yields PCA coordinates. Orthogonal Procrustes aligns the MDS coordinates without translation or scaling; both are already centered. Relative coordinate and pairwise-distance Frobenius discrepancies are below. No PCA routine is used inside mds.py.','',
    '| Dataset | Dimensions | Aligned coordinate error | Pairwise-distance error |','|---|---:|---:|---:|']
    for r in validation:lines.append(f"| {r['dataset']} | {r['dimensions']} | {r['relative_aligned_error']:.3e} | {r['relative_pairwise_error']:.3e} |")
    lines+=['','At q=2, double-centering squared Euclidean distances gives G = X_centered X_centered^T. Its eigenvectors and eigenvalues give the same truncated configuration as centered PCA, up to orthogonal transformations. Reflections and rotations are allowed, but rescaling is not. With eigenvalue ties at a truncation boundary, equivalent choices of subspace may need additional care; the selected numerical comparisons pass.','']
    for slug,s in summaries.items():
        rows=list(s['perq'].values());worst=max(rows,key=lambda r:r['negative_mass_fraction']);raw=max(rows,key=lambda r:r['negative_mass']);unstable=min(rows,key=lambda r:r['mean_neighbor_jaccard'])
        lines += [f'## {slug.title()}: measured behavior','',f"The largest normalized negative mass occurs at q={worst['q']} ({worst['negative_mass_fraction']:.5f}); the largest absolute negative mass occurs at q={raw['q']} ({raw['negative_mass']:.6g}). These are distinct definitions of 'farthest from PSD': the normalized value controls for overall spectral scale, while absolute mass does not. Negative eigenvalues establish that the full dissimilarity geometry is not exactly Euclidean; clipping necessarily discards part of it.",'',f"The lowest mean neighborhood agreement is at q={unstable['q']} ({unstable['mean_neighbor_jaccard']:.5f}); its pairwise-distance rank correlation is {unstable['pairwise_spearman']:.5f}. Compare both measures: overall distance order and local neighborhoods need not change equally. Across non-q=2 values, most-stable rows are {s['most_stable_rows']} (mean Jaccard {s['maximum_stability']:.5f}); least-stable rows are {s['least_stable_rows']} ({s['minimum_stability']:.5f}). These refer to observations, not class labels; detailed per-q scores are in `{slug}_neighborhood_stability.csv`.",'','| Feature | SD | Range | Share of squared pairwise differences | Zero fraction |','|---|---:|---:|---:|---:|']
        for f in s['features']: lines.append(f"| {f['feature']} | {f['sd']:.6g} | {f['range']:.6g} | {f['squared_difference_share']:.5f} | {f['zero_fraction']:.5f} |")
        lines+=['',f"The largest q=2 contribution is {s['features'][0]['feature']} ({s['features'][0]['squared_difference_share']:.2%} of total squared pairwise differences). This contribution uses differences/variances, not raw means: a feature can have a large mean and little influence on distances.",'']
    lines+=['## Units, interpretation and cross-dataset conclusions','',
    'Wine measures different chemical quantities. The archived official description lists their names but does not specify all physical units. In the supplied numerical encoding, Proline dominates squared differences; this makes near-one-dimensional distance structure plausible. This is a property of raw numeric units, not evidence that Proline is intrinsically the most important chemical feature. A change of its units could materially change all q results.','',
    'Seeds mixes area, perimeter/length measurements, asymmetry and derived dimensionless compactness. UCI does not state the physical length/area units on the inspected page, so none are invented here. Area has the largest numerical variation; correlated size features can reinforce one another at low q. The experiment is reproducible in the supplied encoding, but combining these quantities without scaling has no unit-invariant physical interpretation. Compactness C=4πA/P² establishes redundancy independently of our numerical results.','',
    'Glass combines a dimensionless refractive index with eight oxide weight percentages. The chemical components share a reported unit, making their relative magnitudes more directly interpretable than in Wine/Seeds, but refractive index has a very different scale. Compositional constraints also induce dependence. Zero oxide values are documented numeric measurements, not missing-value markers. Calcium and Magnesium variation should be assessed from the feature table rather than the much larger mean Silicon concentration.','',
    'The tables establish which effects are shared: every q=2 spectrum has no meaningful negative eigenvalues, and every q=2 PCA comparison agrees to floating-point precision. For these selected matrices, moving away from q=2 introduces negative spectrum and changes at least some local neighborhoods. Neither the magnitude of non-Euclidean distortion nor the ordering of its worst q is a theorem that applies to every dataset. Compare normalized negative mass, retained positive mass and distance errors separately; no single diagnostic captures them all.','',
    'Mathematically, for a fixed difference vector, ||v||_1 ≥ ||v||_1.5 ≥ ||v||_2 ≥ ||v||_3 ≥ ||v||_4 ≥ ||v||_∞. This does not preserve the ranking of different pairs. Large q emphasizes the largest coordinate difference in each pair; small q accumulates contributions across coordinates. These facts suggest mechanisms for observed neighborhood changes, but do not prove causal explanations of dataset-specific patterns. q=2 guarantees a PSD centered Gram matrix for Euclidean feature distances; other q values need not produce negative eigenvalues on every finite configuration.','',
    '## Visual evidence and limits','',
    'Raw CLI plots for all q and both dimensions are retained under stage4/ as reproducible outputs. They share point style, panel size, equal-axis convention and row order, and identify dataset/q. MDS axes can rotate or reflect independently across q; visual orientation is not itself a geometry change. The numerical tables and rotation-invariant checks are the primary evidence. No class labels were used. Supporting comparison panels, if included, are discussed separately below.','',
    'These conclusions concern exactly three preselected datasets in their original units. They do not establish universal best q values, label separation, or robustness to feature rescaling. Neighborhood changes measured above are in input-distance geometry, not an assertion that every change is visually apparent in 2D. Low-dimensional plots omit positive spectral mass and discard negative spectrum; apparent overlap can therefore be projection error.']
    (ROOT/'results/q_analysis.md').write_text('\n'.join(lines)+'\n'+(ROOT/'code/interpretation.md').read_text())

if __name__=='__main__':main()
