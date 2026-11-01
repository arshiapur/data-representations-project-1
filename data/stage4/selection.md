# Stage 4 selection (locked before q experiments)

Exactly Wine, Seeds and Glass Identification were selected from official UCI pages and inspected archives. No q-dependent results were available at selection. No replacements were made.

Wine: 178 x 13, heterogeneous chemistry with scale imbalance. Seeds: 210 x 7, geometric features with built-in relationships and mixed magnitudes. Glass Identification: 214 x 9, oxide compositions with zeros plus refractive index. These contrasts were selected on documented feature properties, not embedding appearance. All have at least three continuous numeric features after exclusion. Each largest dense 214 x 214 float64 array is 366,368 bytes; pairwise difference arrays are under 5 MB; eigendecomposition costs O(n^3) with n<=214. This is comfortably feasible on this CPU runtime.

Source details, exact files, hashes, dimensions, exclusions, duplicate counts, units and missing policies are in acquisition.json. All rows are retained, no imputation, centering or scaling is applied to saved matrices.
