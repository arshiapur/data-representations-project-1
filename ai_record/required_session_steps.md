# Required AI session records still pending

These records cannot be manufactured from the current Codex conversation. The current tools do not open a separate ChatGPT product session or a fresh user-facing Codex task. A subagent or a Python subprocess would not establish either requirement.

## Stage 1: ChatGPT

Open a new ChatGPT conversation with browsing and file creation available. Submit the entire contents of prompts/Prompt1.txt followed by a newline and Iris. Save the actual response and downloaded Iris.csv. Export the conversation or retain a share link in ai_record/. Compare its source selection, headers, row order and numeric values with data/Iris.csv; document any differences before replacing any matrix. In particular, UCI iris.data and bezdekIris.data can differ, so source choice must be recorded rather than silently reconciled.

## Stage 3: fresh Codex task

Create a fresh Codex task, attach documentation/MDS.pdf and make data/Iris.csv available. Submit the complete prompts/Prompt3.txt. Retain the generated code, actual transcript/task link, test output and both CLI outputs. Compare its distance/Gram construction, eigenvalue policy, filename behavior and q=2 configurations against this submission. Compare configurations by pairwise distances or orthogonal alignment, allowing signs/rotations but no scaling. Document differences and any student edits. Do not claim this step complete until the actual task is run.

## Independent hidden-query check

Optionally have another person pick an undisclosed eligible UCI query, execute Prompt1 in ChatGPT and Prompt2 in Codex without dataset-specific prompt edits, and retain both sessions and outputs. The included Wisconsin diagnostic trial is a portability check in the existing conversation, not a blind test or a guarantee for every query.

## Execution-limit note

The reproduction runner launches 36 separate Python/plotting processes for Stage 4. A short command timeout may end observation before the pipeline completes. Current clean-run exit status, elapsed time and console log are in verification/. It was tested on the cloud machine, not the student computer.
