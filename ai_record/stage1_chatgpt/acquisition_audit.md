# ChatGPT Stage 1: Iris acquisition audit

- Query: `Iris`
- Official UCI dataset: **Iris** (ID 53), https://archive.ics.uci.edu/dataset/53/iris
- Selection rationale: exact match to query.
- Original archive URL: https://archive.ics.uci.edu/static/public/53/iris.zip
- Exact source file: `iris.data` inside the official-source ZIP retained from the prior acquisition run; `iris.names` was consulted.
- **Retrieval limitation:** this ChatGPT execution verified the current official UCI dataset page and raw-text source on the web, but the container could not directly download the UCI ZIP. The CSV was regenerated from the previously retained UCI archive, after verifying its SHA-256 against the recorded acquisition audit. Do not represent this as a fresh live ZIP download.
- Raw shape: 150 flower observations, five columns (4 continuous numeric features plus the species class).
- Output shape: 150 rows x 4 numeric columns, in original row order.
- Features: sepal length, sepal width, petal length, petal width (all in centimeters).
- Excluded: 1 categorical species target column; no ID columns or invented categorical encoding.
- Missing values: documented none; observed none. Removed 0 rows and 0 numeric feature columns. No imputation or scaling.
- Duplicate feature rows: 3, retained.
- Output: `Iris.csv`; byte values preserve numeric measurements.
- Verification: feature matrix agrees exactly with previous submission `data/Iris.csv`.
