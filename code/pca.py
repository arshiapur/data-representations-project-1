"""Deterministic centered/uncentered PCA; run with --input data/STEM.csv."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import scipy
from scipy.linalg import svd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_matrix(path):
    with Path(path).open(newline='') as f:
        rows = csv.reader(f)
        names = next(rows)
        x = np.asarray([[float(v) for v in row] for row in rows], dtype=float)
    if x.ndim != 2 or min(x.shape) < 3 or len(names) != x.shape[1]:
        raise ValueError('Need a header and at least three observations and features')
    if not np.isfinite(x).all():
        raise ValueError('Input must contain finite numeric features only')
    return x


def plot(z, path, title):
    k = z.shape[1]
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d' if k == 3 else None)
    ax.scatter(*z.T, s=15, color='steelblue', alpha=.8)
    for i, setter in enumerate([ax.set_xlabel, ax.set_ylabel] + ([ax.set_zlabel] if k == 3 else [])):
        setter(f'PC{i+1}')
    # Equal numeric limits and physical box dimensions, without changing coordinates.
    mid = (z.min(axis=0) + z.max(axis=0)) / 2
    half = max(float(np.ptp(z, axis=0).max()) * .55, 1e-12)
    for i, setter in enumerate([ax.set_xlim, ax.set_ylim] + ([ax.set_zlim] if k == 3 else [])):
        setter(mid[i]-half, mid[i]+half)
    if k == 3:
        ax.set_box_aspect((1, 1, 1))
    else:
        ax.set_aspect('equal', adjustable='box')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output-dir', default='results', type=Path)
    args = parser.parse_args()
    x = read_matrix(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {'shape': list(x.shape), 'versions': {'numpy': np.__version__, 'scipy': scipy.__version__, 'matplotlib': matplotlib.__version__}, 'routine': 'scipy.linalg.svd(full_matrices=False, lapack_driver=gesdd)', 'scaled': False}
    for method in ['upca', 'cpca']:
        a = x-x.mean(axis=0) if method == 'cpca' else x.copy()
        _, s, vt = svd(a, full_matrices=False, lapack_driver='gesdd')
        z = a @ vt[:3].T
        # Independent NumPy SVD: compare configurations through their Gram matrices,
        # avoiding arbitrary signs (and rotations within equal-eigenvalue spaces).
        _, check_s, check_vt = np.linalg.svd(a, full_matrices=False)
        check_z = a @ check_vt[:3].T
        gram_error = float(np.linalg.norm(z@z.T-check_z@check_z.T)/max(np.linalg.norm(check_z@check_z.T), np.finfo(float).eps))
        np.testing.assert_allclose(s, check_s, rtol=1e-10, atol=1e-10)
        if gram_error > 1e-9:
            raise AssertionError(f'SVD verification discrepancy: {gram_error}')
        mean_norm = float(np.linalg.norm(z.mean(axis=0)))
        if method == 'cpca' and mean_norm > 1e-10*max(1., np.linalg.norm(a)):
            raise AssertionError('Centered projection has nonzero mean')
        report[method] = {'singular_values': s[:3].tolist(), 'projected_mean_norm': mean_norm, 'relative_gram_error': gram_error}
        for k in [2, 3]:
            plot(z[:, :k], args.output_dir/f'{args.input.stem}_{method}_{k}D.png', f'{args.input.stem}: {"Centered" if method == "cpca" else "Uncentered"} PCA ({k}D)')
    (args.output_dir/f'{args.input.stem}_pca_verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
