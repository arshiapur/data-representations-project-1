"""Reconstruct the documented development/selected matrices from retained UCI archives.
This replay script is dataset-specific; Prompt1.txt, rather than this script, handles new queries.
"""
from pathlib import Path
import csv, hashlib, json, zipfile
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
SPECS = [
 ('iris',53,'Iris','iris','iris.data', 'sepal_length,sepal_width,petal_length,petal_width'.split(','),slice(0,4),'Flower; four lengths in cm; exclude species target (column 5). The Iris query exactly matches the official title. Use iris.data as supplied, not bezdekIris.data; retain documented historical discrepancies without manual correction.'),
 ('wine',109,'Wine','wine','wine.data','alcohol,malic_acid,ash,alcalinity_of_ash,magnesium,total_phenols,flavanoids,nonflavanoid_phenols,proanthocyanins,color_intensity,hue,od280_od315,proline'.split(','),slice(1,None),'Wine sample; 13 chemical measurements; exclude cultivar target (column 1). Heterogeneous quantities and substantial scale imbalance. Exact per-feature physical units are not supplied in wine.names; do not infer them.'),
 ('seeds',236,'Seeds','seeds','seeds_dataset.txt','area,perimeter,compactness,kernel_length,kernel_width,asymmetry_coefficient,groove_length'.split(','),slice(0,7),'Wheat kernel; seven documented geometric measurements; exclude variety target (column 8). Compactness is 4*pi*A/P^2 and dimensionless; physical units of area/length are not stated on the UCI page. Geometric redundancy and mixed magnitudes contrast with Wine.'),
 ('glass',42,'Glass Identification','glass+identification','glass.data','refractive_index,sodium,magnesium,aluminum,silicon,potassium,calcium,barium,iron'.split(','),slice(1,10),'Glass sample; exclude ID (column 1) and glass-type target (column 11). Refractive index is dimensionless; eight chemical features are weight percent in corresponding oxide. Compositional dependence and zero concentrations contrast with Wine/Seeds.')]

def main():
 records=[]
 for slug,uid,title,urlslug,member,names,cols,note in SPECS:
  archive=ROOT/'data/sources'/f'{slug}_source.zip'
  with zipfile.ZipFile(archive) as z:
   text=z.read(member).decode()
   if slug=='seeds': raw=[s.split() for s in text.splitlines() if s.strip()]
   else: raw=list(csv.reader(s for s in text.splitlines() if s.strip()))
  # Documentation says no missing values for these four datasets. Reject rather than silently impute.
  x=np.array([[float(v) for v in row[cols]] for row in raw])
  if not np.isfinite(x).all() or x.shape[1]!=len(names): raise ValueError(f'Invalid {slug}')
  out=ROOT/'data/Iris.csv' if slug=='iris' else ROOT/'data/stage4'/f'{slug}.csv'
  out.parent.mkdir(parents=True,exist_ok=True)
  with out.open('w',newline='') as f:
   w=csv.writer(f); w.writerow(names); w.writerows(x)
  records.append(dict(slug=slug,title=title,uci_id=uid,page_url=f'https://archive.ics.uci.edu/dataset/{uid}/{urlslug}',archive_url=f'https://archive.ics.uci.edu/static/public/{uid}/{urlslug}.zip',source_file=member,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),matrix_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),matrix=str(out.relative_to(ROOT)),shape=list(x.shape),features=names,note=note,missing_policy='Official documentation: none; reject unexpected missing/nonfinite values',rows_removed=0,feature_columns_removed=0,excluded_columns=1 if slug!='glass' else 2,duplicate_rows=int(len(x)-len(np.unique(x,axis=0))),row_order='source order; duplicates retained'))
 (ROOT/'data/acquisition.json').write_text(json.dumps(records,indent=2)+'\n')
 # This selection record is written BEFORE any q-dependent computations.
 selection='''# Stage 4 selection (locked before q experiments)\n\nExactly Wine, Seeds and Glass Identification were selected from official UCI pages and inspected archives. No q-dependent results were available at selection. No replacements were made.\n\nWine: 178 x 13, heterogeneous chemistry with scale imbalance. Seeds: 210 x 7, geometric features with built-in relationships and mixed magnitudes. Glass Identification: 214 x 9, oxide compositions with zeros plus refractive index. These contrasts were selected on documented feature properties, not embedding appearance. All have at least three continuous numeric features after exclusion. Each largest dense 214 x 214 float64 array is 366,368 bytes; pairwise difference arrays are under 5 MB; eigendecomposition costs O(n^3) with n<=214. This is comfortably feasible on this CPU runtime.\n\nSource details, exact files, hashes, dimensions, exclusions, duplicate counts, units and missing policies are in acquisition.json. All rows are retained, no imputation, centering or scaling is applied to saved matrices.\n'''
 (ROOT/'data/stage4/selection.md').write_text(selection)
 print(json.dumps([{k:d[k] for k in ['slug','shape','duplicate_rows']} for d in records]))
if __name__=='__main__': main()
