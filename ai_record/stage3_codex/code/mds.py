"""Explicit classical MDS following MDS.pdf, equation (6) and steps 2–3."""
import argparse
import csv
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_q(value):
    if value == 'inf':
        return np.inf, 'inf'
    try:
        d = Decimal(value)
        q = float(d)
    except (InvalidOperation, ValueError, OverflowError):
        raise argparse.ArgumentTypeError('q must be a finite real >=1 or literal inf')
    if not d.is_finite() or d < 1 or not np.isfinite(q):
        raise argparse.ArgumentTypeError('q must be a finite real >=1 or literal inf')
    # Decimal preserves the supplied real number for filename normalization.
    token = format(d, 'f')
    if '.' in token:
        token = token.rstrip('0').rstrip('.')
    return q, token


def read_features(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        rows = list(csv.reader(f))
    if not rows or not rows[0]:
        raise ValueError('empty CSV')
    def numeric(row):
        try:
            return [float(v) for v in row]
        except ValueError:
            return None
    # Optional header only when every cell of the first row is nonnumeric.
    first = numeric(rows[0])
    if first is None:
        for cell in rows[0]:
            try:
                float(cell)
            except ValueError:
                continue
            raise ValueError('mixed numeric/nonnumeric first row')
        rows = rows[1:]
    if not rows or any(len(r) != len(rows[0]) or not r for r in rows):
        raise ValueError('CSV must be a nonempty rectangular numeric matrix')
    try:
        x = np.asarray([[float(v) for v in r] for r in rows], dtype=float)
    except ValueError:
        raise ValueError('all feature cells must be numeric; labels are unsupported')
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1 or not np.isfinite(x).all():
        raise ValueError('need at least two rows, one feature, and finite values')
    return x


def distances(x, q):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or min(x.shape) < 1 or not np.isfinite(x).all():
        raise ValueError('features must be a finite nonempty matrix')
    if not (q == np.inf or np.isfinite(q) and q >= 1):
        raise ValueError('invalid q')
    n = len(x)
    out = np.zeros((n, n))
    with np.errstate(over='raise', invalid='raise', under='ignore'):
        for i in range(n):
            delta = np.abs(x[i+1:] - x[i])
            scale = delta.max(axis=1)
            if q == np.inf:
                values = scale
            else:
                ratios = np.divide(delta, scale[:, None], out=np.zeros_like(delta), where=scale[:, None] != 0)
                # Scaling prevents overflow of delta**q, including very large q.
                values = scale * np.power(np.sum(np.power(ratios, q), axis=1), 1/q)
            out[i, i+1:] = values
            out[i+1:, i] = values
    if not np.isfinite(out).all():
        raise ValueError('distances exceed floating-point range')
    return out


def classical_mds(d, k):
    d = np.asarray(d, dtype=float)
    if k not in (2, 3):
        raise ValueError('dimensions must be 2 or 3')
    if d.ndim != 2 or d.shape[0] != d.shape[1] or len(d) < 2:
        raise ValueError('distance matrix must be square with at least two rows')
    if not np.isfinite(d).all() or (d < 0).any() or not np.allclose(d, d.T) or not np.allclose(np.diag(d), 0):
        raise ValueError('invalid distance matrix')
    with np.errstate(over='raise', invalid='raise'):
        a = d*d
        # Mean after division avoids overflow in an otherwise representable sum.
        means = (a / len(a)).sum(axis=1)
        grand = (means / len(a)).sum()
        g = -0.5*a + 0.5*means[:, None] + 0.5*means[None, :] - 0.5*grand
        g = 0.5*g + 0.5*g.T
    eigenvalues, vectors = np.linalg.eigh(g)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, vectors = eigenvalues[order], vectors[:, order]
    spectral_scale = float(np.max(np.abs(eigenvalues)))
    tol = 10 * len(d) * np.finfo(float).eps * spectral_scale
    positive = eigenvalues > tol
    chosen = np.flatnonzero(positive)[:k]
    y = np.zeros((len(d), k))
    y[:, :len(chosen)] = vectors[:, chosen] * np.sqrt(eigenvalues[chosen])
    mass = float(eigenvalues[positive].sum())
    negative = eigenvalues < -tol
    all_negative = eigenvalues < 0
    diagnostics = {
        'eigenvalues': eigenvalues.tolist(), 'eigenvalue_tolerance': tol,
        'tolerance_formula': '10*n*float64_epsilon*max(abs(eigenvalues)); zero for zero spectrum',
        'positive_eigenvalue_count': int(positive.sum()),
        'negative_count': int(negative.sum()),
        'negative_mass': float(-eigenvalues[negative].sum()),
        'raw_negative_count': int(all_negative.sum()),
        'raw_negative_mass': float(-eigenvalues[all_negative].sum()),
        'positive_mass_fraction_2D': float(eigenvalues[np.flatnonzero(positive)[:2]].sum()/mass) if mass else None,
        'positive_mass_fraction_3D': float(eigenvalues[np.flatnonzero(positive)[:3]].sum()/mass) if mass else None,
        'retained_dimensions': len(chosen), 'zero_padded_dimensions': k-len(chosen),
    }
    # Normalize before squaring, avoiding overflow in sums of squared distances.
    dhat = distances(y, 2)
    upper = np.triu_indices(len(d), 1)
    scale = float(d.max())
    if scale:
        target, fitted = d[upper]/scale, dhat[upper]/scale
        diagnostics['normalized_distance_rmse'] = float(np.linalg.norm(fitted-target)/np.linalg.norm(target))
        diagnostics['normalized_squared_distance_rmse'] = float(np.linalg.norm(fitted**2-target**2)/np.linalg.norm(target**2))
    else:
        diagnostics['normalized_distance_rmse'] = 0.0
        diagnostics['normalized_squared_distance_rmse'] = 0.0
    diagnostics['error_definitions'] = {
        'normalized_distance_rmse': 'sqrt(sum_i<j (dhat_ij-d_ij)^2 / sum_i<j d_ij^2)',
        'normalized_squared_distance_rmse': 'sqrt(sum_i<j (dhat_ij^2-d_ij^2)^2 / sum_i<j d_ij^4)',
        'zero_denominator': '0 for all-zero distances and embedding',
    }
    return y, diagnostics


def plot_coordinates(y, path):
    k = y.shape[1]
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d' if k == 3 else None)
    ax.scatter(*y.T, s=22, color='#2468a2', alpha=0.8)
    ax.set_xlabel('MDS coordinate 1')
    ax.set_ylabel('MDS coordinate 2')
    if k == 2:
        ax.set_aspect('equal', adjustable='box')
    else:
        ax.set_zlabel('MDS coordinate 3')
        center = (y.min(axis=0)+y.max(axis=0))/2
        radius = max(float(np.ptp(y, axis=0).max())/2, 0.5)
        ax.set_xlim(center[0]-radius, center[0]+radius)
        ax.set_ylim(center[1]-radius, center[1]+radius)
        ax.set_zlim(center[2]-radius, center[2]+radius)
        ax.set_box_aspect((1, 1, 1))
    ax.set_title(path.stem)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--q', required=True, type=parse_q)
    parser.add_argument('--dimensions', required=True, type=int, choices=(2, 3))
    args = parser.parse_args()
    try:
        if args.input.suffix.lower() != '.csv':
            raise ValueError('input must have a .csv extension')
        x = read_features(args.input)
        q, token = args.q
        y, report = classical_mds(distances(x, q), args.dimensions)
        stem = f'{args.input.stem}_mds_q{token}_{args.dimensions}D'
        csv_path, png_path = Path(stem+'.csv'), Path(stem+'.png')
        np.savetxt(csv_path, y, delimiter=',', header=','.join(f'MDS{i+1}' for i in range(args.dimensions)), comments='', fmt='%.17g')
        plot_coordinates(y, png_path)
        report.update(input=str(args.input), rows=len(x), features=x.shape[1], q=token, dimensions=args.dimensions, coordinate_csv=str(csv_path), plot_png=str(png_path))
        print(json.dumps(report, allow_nan=False))
    except (ValueError, OSError, FloatingPointError, np.linalg.LinAlgError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
