#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generated from swiss_roll_1.ipynb.
This script reproduces the corresponding numerical experiment in the manuscript.
"""
# ============================================================
# Section 5.2: Swiss roll with density removal and dimension gating
# Combined code for the input figure and the numerical experiment
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from numpy.linalg import eigh
from scipy.spatial.distance import pdist
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed

try:
    import pandas as pd
except ImportError:
    pd = None


# ---------------- Global settings ----------------
N_MANIFOLD = 800
T_LO, T_HI = 1.5 * np.pi, 4.5 * np.pi
HEIGHT = 21.0

# Clutter levels used in the numerical experiment.
CLUTTER_LEVELS = [0, 50, 100, 200, 400]

# Number of repetitions and parallel jobs.
# Increase N_REP for the final run if needed.
N_REP = 8
N_JOBS = 7

# Graph and local-dimension parameters.
K_GRAPH = 15        # Number of nearest neighbors in the base graph
K_H = 5             # Neighbor rank used to estimate the kernel bandwidth
K_DIM = 20          # Maximum neighbor count used for local dimension estimation
DIM_KS = [10, 15, 20]

# Dimension-gate parameters.
M0 = 2.0            # Intrinsic dimension of the clean Swiss roll
ETA = 2.0           # Sharpness of the local-dimension gate

# Diffusion-map parameters.
R_EMB = 2           # Embedding dimension
T_DIFF = 1.0        # Diffusion time

# Figure settings.
PLOT_SEED = 1400
PLOT_CLUTTER = 400
OUTDIR = "gdc_figures"
os.makedirs(OUTDIR, exist_ok=True)

EPS = 1e-12

# %%
# ---------------- Data generation ----------------
def arclength(t):
    """Compute the arc-length coordinate of the planar spiral r=t."""
    return 0.5 * (t * np.sqrt(1.0 + t * t) + np.arcsinh(t))


def make_swiss(seed, n_manifold=N_MANIFOLD, return_parameters=False):
    """
    Generate a clean Swiss roll in R^3.

    The ground-truth intrinsic coordinates are (arc length, height), which are
    used only for evaluating preservation of the manifold geometry.
    """
    rng = np.random.default_rng(seed)
    t = rng.uniform(T_LO, T_HI, n_manifold)
    h = rng.uniform(0.0, HEIGHT, n_manifold)

    X = np.column_stack([
        t * np.cos(t),
        h,
        t * np.sin(t)
    ])

    truth = np.column_stack([arclength(t), h])

    if return_parameters:
        return X, truth, t, h
    return X, truth


def add_clutter(X, n_clutter, seed):
    """
    Add uniform off-manifold clutter points in the bounding box of X.

    The returned mask marks the original manifold points.  Evaluation is always
    performed on the manifold points only, while the graph is built on all
    points, including clutter.
    """
    if n_clutter == 0:
        return X.copy(), np.ones(len(X), dtype=bool), np.empty((0, X.shape[1]))

    rng = np.random.default_rng(seed + 10_000)
    lo, hi = X.min(axis=0), X.max(axis=0)
    C = rng.uniform(lo, hi, size=(n_clutter, X.shape[1]))

    Xall = np.vstack([X, C])
    mask = np.zeros(len(Xall), dtype=bool)
    mask[:len(X)] = True

    return Xall, mask, C


# ---------------- Graph construction and auxiliary quantities ----------------
def base_kernel(Xall, h):
    """
    Build a symmetric kNN Gaussian affinity matrix.

    The same base graph is used by all methods; only the normalization or
    gating step changes.
    """
    nn = NearestNeighbors(n_neighbors=K_GRAPH + 1).fit(Xall)
    dist, idx = nn.kneighbors(Xall)

    n = len(Xall)
    W = np.zeros((n, n))
    weights = np.exp(-(dist[:, 1:] ** 2) / (2.0 * h * h))

    for i in range(n):
        W[i, idx[i, 1:]] = weights[i]

    # Symmetrize the directed kNN graph.
    W = np.maximum(W, W.T)
    return W


def bandwidth(Xman):
    """
    Estimate a global bandwidth from the clean manifold scale.

    This keeps the base kernel scale fixed across clutter levels.
    """
    nn = NearestNeighbors(n_neighbors=K_H + 1).fit(Xman)
    dist, _ = nn.kneighbors(Xman)
    return np.median(dist[:, K_H])


def local_dim(Xall):
    """
    Estimate local intrinsic dimension by the Levina--Bickel MLE.

    Several neighbor counts are averaged for stability.
    """
    nn = NearestNeighbors(n_neighbors=K_DIM + 1).fit(Xall)
    dist, _ = nn.kneighbors(Xall)

    # Remove self-distances and protect against zero distances.
    dist = np.maximum(dist[:, 1:], EPS)

    estimates = []
    for k2 in DIM_KS:
        Tk = dist[:, k2 - 1][:, None]
        logs = np.log(Tk / dist[:, :k2 - 1])
        inv_dim = logs.sum(axis=1) / (k2 - 1)
        estimates.append(1.0 / np.maximum(inv_dim, 1e-9))

    return np.mean(estimates, axis=0)


# ---------------- Diffusion embedding ----------------
def diffusion_embed(W, r=R_EMB, t=T_DIFF):
    """
    Compute the diffusion-map embedding from a symmetric affinity matrix.
    """
    d = W.sum(axis=1) + EPS
    sq = np.sqrt(d)

    # Symmetric conjugate of the random-walk matrix.
    S = W / sq[:, None] / sq[None, :]

    val, vec = eigh(S)
    order = np.argsort(val)[::-1]
    val, vec = val[order], vec[:, order]

    # Right eigenvectors of P = D^{-1} W.
    phi = vec / sq[:, None]

    # Remove the trivial eigenvector.
    return phi[:, 1:1 + r] * (val[1:1 + r] ** t)


# ---------------- Affinity transformations ----------------
def affinity_DM(K, alpha):
    """
    Coifman--Lafon alpha normalization.

    alpha=1 removes the sampling-density drift in the usual diffusion-map
    continuum limit.
    """
    d = K.sum(axis=1) + EPS
    return K / (d[:, None] ** alpha) / (d[None, :] ** alpha)


def affinity_GD(K, q, gamma):
    """
    Gamma-Dirichlet density gate based on the degree-density q.
    """
    gate = (q[:, None] * q[None, :]) ** gamma
    return gate * K


def affinity_gate_then_DM(K, w, gamma):
    """
    Counterexample order: apply the dimension gate first, then alpha=1.

    The post-gate degree normalization can partially undo the clutter
    suppression, so this order is included as a diagnostic contrast.
    """
    C = ((w[:, None] * w[None, :]) ** gamma) * K
    d_post = C.sum(axis=1) + EPS
    return C / d_post[:, None] / d_post[None, :]


def affinity_DG(K, w, gamma):
    """
    Proposed dimension-gated affinity: alpha=1 density removal, then dimension gate.

    The degree d removes sampling-density effects on the manifold, whereas the
    independent gate w suppresses off-manifold clutter.
    """
    d = K.sum(axis=1) + EPS
    K_alpha = K / d[:, None] / d[None, :]
    gate = (w[:, None] * w[None, :]) ** gamma
    return gate * K_alpha

# %%
# ---------------- Figure: clean versus contaminated Swiss roll ----------------
def truncate_colormap(cmap_name="viridis", minval=0.05, maxval=0.78, n=256):
    """
    Truncate a colormap to avoid the bright yellow end of viridis.
    """
    cmap = plt.get_cmap(cmap_name)
    return LinearSegmentedColormap.from_list(
        f"{cmap_name}_truncated",
        cmap(np.linspace(minval, maxval, n))
    )


def plot_clean_vs_clutter(
    seed=PLOT_SEED,
    n_clutter=PLOT_CLUTTER,
    outdir=OUTDIR,
    clutter_color="#cc00cc",
):
    """
    Save the side-by-side input-data figure used in Section 5.2.
    """
    Xman, truth, roll_t, height = make_swiss(seed, return_parameters=True)
    Xall, mask, Xclutter = add_clutter(Xman, n_clutter, seed)

    # Use common axis limits for the two panels.
    mins = Xall.min(axis=0)
    maxs = Xall.max(axis=0)
    centers = (mins + maxs) / 2.0
    radius = 0.5 * np.max(maxs - mins)

    xlim = (centers[0] - radius, centers[0] + radius)
    ylim = (centers[1] - radius, centers[1] + radius)
    zlim = (centers[2] - radius, centers[2] + radius)

    cmap_roll = truncate_colormap("viridis", 0.05, 0.78)

    fig = plt.figure(figsize=(11.0, 5.2))
    axes = []
    titles = [
        "(A) Clean Swiss roll",
        "(B) Swiss roll with uniform clutter",
    ]

    for panel in range(2):
        ax = fig.add_subplot(1, 2, panel + 1, projection="3d")
        axes.append(ax)

        sc = ax.scatter(
            Xman[:, 0], Xman[:, 1], Xman[:, 2],
            c=roll_t,
            s=9,
            cmap=cmap_roll,
            alpha=0.92,
            linewidths=0,
            depthshade=False,
        )

        if panel == 1:
            ax.scatter(
                Xclutter[:, 0], Xclutter[:, 1], Xclutter[:, 2],
                c=clutter_color,
                s=13,
                alpha=0.65,
                linewidths=0,
                depthshade=False,
                label="uniform clutter",
            )
            ax.legend(loc="upper left", frameon=True, fontsize=8)

        ax.set_title(titles[panel], pad=8)
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        ax.set_zlabel(r"$x_3$")

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)

        ax.view_init(elev=18, azim=-65)

        # Use faint pane backgrounds for a cleaner manuscript figure.
        ax.xaxis.pane.set_alpha(0.05)
        ax.yaxis.pane.set_alpha(0.05)
        ax.zaxis.pane.set_alpha(0.05)

    cbar = fig.colorbar(sc, ax=axes, shrink=0.72, pad=0.04, fraction=0.035)
    cbar.set_label(r"roll parameter $t$")

    fig.suptitle("Swiss roll data: clean and contaminated cases", y=0.98)
    plt.tight_layout(rect=[0.00, 0.00, 0.95, 0.96])

    pdf_path = os.path.join(outdir, "swiss_roll_clean_vs_clutter400_3d_magenta.pdf")
    png_path = os.path.join(outdir, "swiss_roll_clean_vs_clutter400_3d_magenta.png")

    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.show()

    print("Saved:", pdf_path)
    print("Saved:", png_path)

    return pdf_path, png_path

# %%
# ---------------- Numerical experiment ----------------
def geodesic_rho(coords, truth, mask):
    """
    Correlate pairwise distances in the embedding with true manifold distances.

    The correlation is computed only among the original manifold points.  Clutter
    points affect the graph and the embedding, but are excluded from scoring.
    """
    embedded_dist = pdist(coords[mask])
    true_dist = pdist(truth)
    return np.corrcoef(embedded_dist, true_dist)[0, 1]


def one_run(n_clutter, seed):
    """
    Run one Swiss-roll experiment for a fixed clutter level and random seed.
    """
    Xman, truth = make_swiss(seed)
    Xall, mask, _ = add_clutter(Xman, n_clutter, seed)

    # The bandwidth is tied to the clean manifold scale.
    h = bandwidth(Xman)

    # The base graph is shared by all methods.
    K = base_kernel(Xall, h)

    # Degree-density used by GD.
    q = K.sum(axis=1) / K.sum()

    # Local-dimension gate used by DG.
    m_hat = local_dim(Xall)
    w_dim = np.exp(-ETA * np.maximum(m_hat - M0, 0.0))

    out = {}
    out["DM a=0"] = geodesic_rho(
        diffusion_embed(affinity_DM(K, 0.0)), truth, mask
    )
    out["DM a=1"] = geodesic_rho(
        diffusion_embed(affinity_DM(K, 1.0)), truth, mask
    )
    out["GD g=1.5"] = geodesic_rho(
        diffusion_embed(affinity_GD(K, q, 1.5)), truth, mask
    )
    out["GD g=2.5"] = geodesic_rho(
        diffusion_embed(affinity_GD(K, q, 2.5)), truth, mask
    )
    out["gate-then-DM g=1.5"] = geodesic_rho(
        diffusion_embed(affinity_gate_then_DM(K, w_dim, 1.5)), truth, mask
    )
    out["DG g=1.5"] = geodesic_rho(
        diffusion_embed(affinity_DG(K, w_dim, 1.5)), truth, mask
    )
    out["DG g=2.5"] = geodesic_rho(
        diffusion_embed(affinity_DG(K, w_dim, 2.5)), truth, mask
    )

    return n_clutter, out


def run_experiment(
    clutter_levels=CLUTTER_LEVELS,
    n_rep=N_REP,
    n_jobs=N_JOBS,
    outdir=OUTDIR,
):
    """
    Run all repetitions and save a compact summary table.
    """
    tasks = [
        (n_clutter, 1000 * rep + n_clutter)
        for n_clutter in clutter_levels
        for rep in range(n_rep)
    ]

    results = Parallel(n_jobs=n_jobs)(
        delayed(one_run)(n_clutter, seed) for n_clutter, seed in tasks
    )

    columns = [
        "DM a=0",
        "DM a=1",
        "GD g=1.5",
        "GD g=2.5",
        "gate-then-DM g=1.5",
        "DG g=1.5",
        "DG g=2.5",
    ]

    agg = {c: {k: [] for k in columns} for c in clutter_levels}
    for n_clutter, out in results:
        for key in columns:
            agg[n_clutter][key].append(out[key])

    summary_rows = []
    for n_clutter in clutter_levels:
        row = {"clutter": n_clutter}
        for key in columns:
            values = np.asarray(agg[n_clutter][key])
            row[f"{key} mean"] = values.mean()
            row[f"{key} sd"] = values.std(ddof=0)
        summary_rows.append(row)

    if pd is not None:
        summary = pd.DataFrame(summary_rows)
        csv_path = os.path.join(outdir, "swiss_roll_section52_results.csv")
        summary.to_csv(csv_path, index=False)
        print("Saved:", csv_path)
    else:
        summary = summary_rows

    header = "clutter | " + " | ".join(f"{key:>20}" for key in columns)
    print(header)
    print("-" * len(header))

    for n_clutter in clutter_levels:
        entries = []
        for key in columns:
            values = np.asarray(agg[n_clutter][key])
            entries.append(f"{values.mean():6.3f}({values.std(ddof=0):.2f})")
        print(f"{n_clutter:7d} | " + " | ".join(f"{x:>20}" for x in entries))

    print()
    print("Interpretation:")
    print("  gate-then-DM applies the dimension gate before alpha=1 normalization;")
    print("  this order can undo part of the clutter suppression.")
    print("  DG applies alpha=1 density removal first and the independent dimension")
    print("  gate second; this is the proposed density-removal-plus-gating order.")

    return summary


def main():
    """Generate the manuscript figure and run the Swiss-roll experiment."""
    os.makedirs(OUTDIR, exist_ok=True)
    plot_clean_vs_clutter()
    summary = run_experiment()
    return summary


if __name__ == "__main__":
    main()
