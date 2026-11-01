# Prompt and numerical recheck

This additional check followed the user's request to try the submitted prompts again. It was performed in the current Codex conversation, not by a separately spawned model or an independent ChatGPT session.

## Requirement review

- Prompt1 specifies official UCI search/documentation, adaptive interpretation, only one dataset-specific trailing argument, original filename capitalization/underscores, numeric unsupervised features, removal of labels/IDs/metadata, documented missing handling and honest failure.
- Prompt2 specifies the exact centered and uncentered definitions, established routines/settings, no feature scaling, singular-value/centering checks, fixed row order, no labels and all four equally scaled plots. Its fallback wording was clarified to check for the actual input file rather than the existence of a directory.
- Prompt3 names the attached handout and allowed/prohibited operations, full q support, spectra/diagnostics, tests and exact output naming. Its primary CLI wording was clarified to reproduce the assignment literally: `python mds.py --input dataset.csv --q Q --dimensions K`; code/mds.py remains its archive location.
- Prompt4 specifies exactly three selections locked before q results, eligibility/contrasts/provenance, a common compliant six-value q grid, both dimensions, complete spectral/distance/neighborhood metrics, per-observation stability, numerical PCA validations, units, and bounded interpretation.

No numerical algorithm change was needed during this recheck. Text clarifications were consolidated in the final prompts.

## Non-Iris Prompt1/Prompt2 trial

Query: `wisconsin_cancer_diagnostic`, with no dataset-specific edits to either prompt. Exact concatenated messages are retained in prompt_checks/Prompt1_executed.txt and Prompt2_executed.txt. Official UCI search with all query words was attempted; its static response reported zero results, so it was not presented as a successful search hit. The official matching dataset page and linked archive were reached and inspected directly. The chosen title matches all words and distinguishes the Diagnostic variant from Original and Prognostic.

Official selection: Breast Cancer Wisconsin (Diagnostic), UCI ID 17. Inspected wdbc.names and wdbc.data in the official linked ZIP. Removed ID column 1 and diagnosis column 2. Retained 30 continuous numeric features (ten measurements summarized by mean, standard error and worst), 569 observations, original order and units. Documentation and observed data show no missing features; no imputation, row dropping or scaling was performed. No class labels were used in plots.

All four required PCA images were generated with the original underscore-preserving filename stem. Independent NumPy-versus-SciPy SVD Gram discrepancies were 2.63e-16 (uncentered) and 2.62e-16 (centered). Centered projected mean norm was 4.62e-13 in the original numerical units. The input matrix hash was unchanged after computation.

This is a **supplementary portability check**, not a fourth Stage 4 dataset. Source bytes, documentation, numeric matrix, audit, exact prompt messages, code and output figures are retained under ai_record/prompt_checks/. Reproduce it with `python ai_record/prompt_checks/replay.py`. The acquisition replay code is necessarily dataset-specific after document inspection; the submitted acquisition prompt itself has no hardcoded dataset details. This does not constitute a fresh blind-model evaluation.

## Full project rerun

`python code/reproduce.py` completed successfully again: all five unittest methods passed, all six required Iris figures were regenerated, all 36 Stage 4 CLI runs completed, and all six centered-PCA validations passed. Current actual test output is results/test_results.txt.

An additional independent audit uses scipy.spatial.distance.cdist for all q distances, explicit dense J matrix multiplication for double-centering, and NumPy eigenvalues. It checked spectra, coordinate shapes/centering and recomputed normalized errors for all 36 embeddings. Maximum relative spectrum discrepancy was 2.635e-15; all recorded distance errors agreed to within 1e-12. Run `python code/audit_results.py`; detailed current output is results/independent_audit.json.

## Remaining process limits

The computational checks pass, but no separate ChatGPT product session or fresh Codex task was created by this check. The assignment explicitly requests those execution contexts; README retains the steps for the student to perform them. This record does not replace or misrepresent that requirement.
