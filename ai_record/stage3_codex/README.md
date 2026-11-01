# Stage 3 classical MDS submission

This is the fresh run authorized in the current Codex conversation. The starter ZIP supplied the prompt, MDS handout and Iris data; it supplied no completed results. The full five-page handout was read before implementation. No START_HERE.md is needed in this submission.

## Run

Requires Python 3, NumPy and Matplotlib (versions recorded in results/environment.json).

From the submission root:

```sh
python code/mds.py --input data/Iris.csv --q 2 --dimensions 2
python code/mds.py --input data/Iris.csv --q 2 --dimensions 3
```

Files are written to the current working directory. The retained Iris commands were run from results/ using ../code/mds.py and ../data/Iris.csv so both PNGs and coordinate CSVs reside in results/. The required equivalent command from code/ is `python mds.py --input ../data/Iris.csv --q 2 --dimensions 2`.

`python run_verification.py` reruns the tests, both Iris commands and independent alignment check, saving actual outputs and TASK_EXECUTION_TRANSCRIPT.txt. This transcript is a real subprocess execution record, not a complete platform conversation export. A platform task link/export is not exposed by the available tools. Prompt3.txt preserves the submitted specification.

## Mathematics and assumptions

Each CSV row is an object; all columns are numeric features, unchanged and in original order. An optional first row of wholly nonnumeric column names is accepted. Mixed headers, ragged data, nonnumeric data cells, fewer than two observations and nonfinite values are rejected. Labels, IDs and categorical columns must not be supplied; there is no automatic column selection, scaling or labeling. Iris has 150 rows and four feature columns. CSV output has one header and K numeric columns in input row order. Plot labels name coordinate axes, not individual objects or classes; all points have the same style.

For finite real q>=1, distances use m*(sum((abs(delta)/m)**q))**(1/q), where m=max(abs(delta)); a zero difference gives zero. This prevents power overflow for large q. Literal `inf` gives max-coordinate distance. Computation uses float64: decimal q is converted to the nearest representable float, while filename normalization uses the original decimal value. Unrepresentable distances, squares or Gram arithmetic fail explicitly rather than being silently rescaled. Runtime is O(n^3) and memory O(n^2); this is a dense implementation.

The handout's equation (6) is implemented as G=-0.5*(D²-rowmean-columnmean+grandmean), followed by (G+G.T)/2. Symmetric eigendecomposition is sorted descending. Coordinates are eigenvectors times square roots of the retained eigenvalues; they are not normalized further. This directly implements STRAIN/classical MDS, with no MDS/PCA/kernel-PCA library calls.

Tolerance is tau=10*n*eps64*max(abs(lambda)), or zero for an all-zero spectrum. This scales with Gram units and matrix size and adds a modest safety factor to floating-point eigendecomposition error; it is a numerical heuristic, not a statistical noise threshold. Eigenvalues above tau are positive; values below -tau are meaningful negatives. Values in [-tau,tau] are treated as zero. No fixed absolute floor is imposed, so changing data units does not change numerical rank in ordinary representable arithmetic. Raw eigenvalues, raw negative count/mass and meaningful negative count/mass are all reported. This avoids hiding round-off while distinguishing meaningful non-Euclidean structure. With fewer than K positive values, remaining columns are zero, including when K exceeds n.

Positive mass fractions sum the largest up-to-2 or up-to-3 eigenvalues above tau divided by the total positive mass. They are null if positive mass is zero. Negative mass is the absolute sum of meaningful negative eigenvalues. These quantities describe Gram spectrum, not distance fit.

Distance errors compare Euclidean distances in the fitted coordinates to original Minkowski dissimilarities over unordered distinct pairs. JSON defines both errors explicitly: sqrt(sum((dhat-d)^2)/sum(d^2)) and sqrt(sum((dhat²-d²)^2)/sum(d^4)). An all-zero dissimilarity matrix has both errors zero. Computation normalizes distances before evaluating these ratios for stability. These are diagnostic relative errors; they are not optimized STRESS objectives and do not replace STRAIN.

Finite q filenames use plain decimal notation with no redundant trailing fractional zeros; `2`, `2.0`, `2.00` all give q2. Infinity gives qinf. The input basename loses only its final .csv extension; capitalization and underscores remain. Output names are STEM_mds_qQ_KD.csv/png.

## Verification

Seven automated test methods cover a unit square (Euclidean spectrum 1,1,0,0), the handout's 3-4-5 triangle Gram matrix, Manhattan and max-coordinate distances, fractional and very large q, Manhattan-square negative spectrum (2,2,0,-1), rank deficiency/zero data, CLI validation and output filenames/files.

The separate Iris verification centers the original feature matrix and computes a general SVD to form independent PCA-equivalent coordinates. It aligns only through an orthogonal matrix, without scaling; errors are saved in results/iris_alignment.json. This validation is separate from the MDS implementation and does not substitute PCA for MDS. Eigenvector signs and bases within repeated eigenvalues are not uniquely determined.
