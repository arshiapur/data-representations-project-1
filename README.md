# Dimensionality reduction with ChatGPT and Codex

Iris is the development dataset. The three Stage 4 datasets selected by Codex are **Wine**, **Seeds**, and **Glass Identification**. The analysis is in results/q_analysis.md. Final prompts are in prompts/; the exact required filenames are preserved. All numeric matrices contain features only, with no labels or IDs.

## Reproduce the results

Use Python 3.12 (tested 3.12.14). Required packages and tested versions are numpy 2.3.5, scipy 1.17.0 and matplotlib 3.10.8, pinned in requirements.txt. No randomized algorithm is used in the final results. From this archive's root:

    python3 -m venv .venv
    .venv/bin/python -m pip install -r requirements.txt
    .venv/bin/python code/reproduce.py

This runs the five MDS tests, generates all four required Iris PCA PNGs and both Iris q=2 MDS PNGs, executes all 36 Stage 4 CLI runs (three datasets × six q × two dimensions), numerically validates PCA/MDS in both dimensions, rebuilds the analysis, and generates supporting panels. Exit status is preserved; a failed command stops the run. Logs, spectra, coordinate CSVs, metrics and test results remain in results/. On Windows activate the virtual environment and use `python` instead of `.venv/bin/python`.

For individual steps:

    python code/test_mds.py
    python code/pca.py --input data/Iris.csv --output-dir results
    # MDS writes files in the current working directory:
    cd results
    python ../code/mds.py --input ../data/Iris.csv --q 2 --dimensions 2
    python ../code/mds.py --input ../data/Iris.csv --q 2 --dimensions 3
    cd ..
    python code/investigate.py
    python code/comparison_panels.py

The portable program interface is exactly `python mds.py --input dataset.csv --q Q --dimensions K` when run from its directory (or use the path to the script). Q>=1 or literal inf; K is 2 or 3. Input basename determines output names, with normalized q tags, in the working directory.

## Prompt use and hidden datasets

Prompt filenames below mean **their complete contents**, not the literal filename.

1. Give ChatGPT prompts/Prompt1.txt followed by whitespace and `Iris` in one message. For a hidden query substitute ONLY that final query, preserving capitalization and underscores. The prompt searches official UCI sources, inspects unfamiliar packaging and documentation, excludes targets/IDs/metadata, records treatment and provenance, and produces data/QUERY.csv. It must report retrieval failure honestly rather than substitute data.
2. Give Codex prompts/Prompt2.txt followed by whitespace and the SAME query in one message. PCA uses that matrix directly. Do not supply dataset-specific column names or modify the reusable prompt. code/pca.py can process the resulting numeric matrix generically.
3. The assignment requests a **fresh Codex task**: attach documentation/MDS.pdf and submit prompts/Prompt3.txt. The supplied Multidimensional-Scaling.pdf has been retained under that required handout name. The implementation follows its STRAIN derivation, including equation (6), negative-eigenvalue clipping and top-K coordinate construction. This project's actual implementation and execution took place in one continuing Codex conversation; it is not represented as a separate fresh-task run. If the instructor enforces this process requirement, rerun this consolidated prompt in a fresh task with data/Iris.csv available and compare results.
4. Give Codex prompts/Prompt4.txt with mds.py available and **no dataset argument**. Stage 4 selects its own three eligible UCI datasets before examining q-dependent results. The submitted replay uses the locked selection in data/stage4/selection.md; it does not choose new datasets each replay.

Prompts are retained as consolidated instructions; the artifacts in this archive were generated while following them in this Codex conversation. No independent ChatGPT product session or separate fresh Codex session is claimed. The separate product/session execution requested by the assignment should be completed by the student if required by the instructor.

## Data and source records

See data/acquisition.json for official titles, IDs, URLs, exact archive members, SHA-256 hashes, matrix shapes, stable feature names, exclusions, missing policies, units and duplicate counts. Official UCI pages and original ZIP archives are retained in data/sources/. Targets and metadata exist in those **raw source archives only**, never in analysis CSVs or plots. Preserve these to audit the cleaning decisions. Raw sources are credited to the respective UCI dataset creators; their official pages contain citation/license details (CC BY 4.0).

`python code/prepare_data.py` reconstructs all submitted matrices from the retained official archives, preserving source order and numeric values. It is a dataset-specific replay script based on inspected documentation, not the reusable hidden-query acquisition prompt. It also reproduces the same selection record; initial selection preceded all q-dependent computations. No rows were removed, imputed, normalized or standardized. Duplicates were retained. Iris uses iris.data, including UCI's documented historical discrepancies, without manual correction or replacement by a packaged benchmark.

A clean result run needs no files outside this archive, except package installation and optional retrieval of the explicitly named public UCI sources. The original PDF documents are included. No API keys, background services or class labels are needed. Required downloads need HTTPS access to archive.ics.uci.edu; results can be reproduced offline once dependencies are present.

## Method and validation

PCA uses deterministic scipy.linalg.svd on X or X−mean(X), without standard-deviation scaling. An independent NumPy SVD comparison and mean checks are saved in results/Iris_pca_verification.json. Required PCA points are not rotated or rescaled after projection. All plots use a uniform point style and equal physical axis scales; no class labels influence any result.

Classical MDS implements distances, double-centering, symmetric eigendecomposition, sorted eigenvalues, clipping and coordinates explicitly. Its scale-aware tolerance and all diagnostic definitions are documented in code/mds.py and results/q_analysis.md. Tests include the handout's 3–4–5 triangle and exact Gram matrix, Euclidean and Manhattan squares including a negative eigenvalue, q=1/1.5/inf/large q, duplicates, zero distances and CLI validation/naming/output checks. Five unittest methods passed; assertions cover multiple cases per method.

For each Stage 4 dataset, q=2 agrees with independently computed centered PCA in 2D and 3D after orthogonal alignment without scaling; numerical discrepancies are tabulated in the analysis. These compare the two independently truncated configurations, not a truncated embedding against original full-dimensional distances. MDS distance errors can remain nonzero even at q=2.

Supporting comparison panels use rotation/reflection solely for visual orientation and common axes per dataset. They are not the required PCA plots; no coordinates are rescaled. Their interpretation distinguishes norm contraction, shape changes, input-neighborhood changes and projection artifacts.

## AI records and code changes

See ai_record/ for execution records, change history and verification evidence. All code and text were AI-generated and revised in this conversation. No human manual code changes are reported. The student should retain/export this conversation as the original interaction record and review every submitted claim and line of code. The archive includes relevant summarized AI decisions and actual command output; it does not claim to include a complete platform transcript.

Use the existing checkout in cloud tasks, which are already isolated; no additional Git worktree is needed. In the cloud runtime, writable plotting caches are set by the runner. No environment configuration was republished by the agent after the user's publication.

## Additional prompt recheck

The prompts were reviewed again and tried on `wisconsin_cancer_diagnostic` in the current Codex session. A 569×30 feature matrix and all four PCA figures were produced, with independent SVD discrepancies below 3e-16. This supplementary portability dataset is **not** part of the three Stage 4 selections. See ai_record/prompt_recheck.md for scope, exact prompt messages and evidence; `python ai_record/prompt_checks/replay.py` reproduces that check. `python code/audit_results.py` independently verifies all 36 Stage 4 embeddings against SciPy distances and a separate Gram/spectrum calculation. Separate ChatGPT/fresh-Codex execution remains a student process step.

## Clean reproduction confirmation

A fresh output directory containing no existing results completed `python code/reproduce.py` with exit status 0 in 69.10 seconds on the current cloud machine. It regenerated six required Iris PNGs and all 36 Stage 4 PNGs. Both standalone Stage 3 CLI commands also exited 0 and produced the expected CSV/PNG files in their current directory. All 36 regenerated embeddings passed an independent numerical audit and matched the submitted coordinates. Actual console logs and machine-readable status are retained in ai_record/verification/. This verifies a clean result regeneration in the cloud runtime, not execution on the student computer. See ai_record/required_session_steps.md for the still-pending ChatGPT and fresh-Codex transcript requirements.

## User-supplied Stage 1 follow-up

The user subsequently supplied a ChatGPT acquisition audit and Iris.csv, preserved verbatim in ai_record/stage1_chatgpt/. Independent comparison found identical feature headers, row order and all 150×4 numeric values relative to data/Iris.csv, so no matrix or downstream result was replaced. The supplied audit reports that ChatGPT verified the current UCI page/raw-text source but regenerated the CSV from the previously retained official ZIP because a fresh ZIP download was unavailable. This is not claimed as fresh live archive retrieval. Earlier notes about no separate ChatGPT artifacts describe the original run; these follow-up artifacts are now included. The actual ChatGPT conversation export/share link has not yet been supplied, and the fresh Stage 3 Codex task remains pending.
