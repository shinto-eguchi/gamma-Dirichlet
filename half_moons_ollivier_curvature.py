"""Half-moons Ollivier--Ricci curvature diagnostic for Gamma--Dirichlet geometry.

Paper setting:
    noise=0.18, n_clutter=0, seed=0, gamma=(0, 3, 5), kNN=15.
The edge loop is parallelized with joblib (default: 7 workers).

This script is the non-notebook counterpart of half_moons_ollivier_curvature.ipynb.
"""

import os
# Important: set BLAS/OpenMP threads before NumPy/SciPy imports.
# The outer edge loop is parallelized with joblib.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from functools import lru_cache

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import linprog
from scipy.sparse import csr_matrix, diags, triu
from scipy.sparse.csgraph import shortest_path

from sklearn.datasets import make_moons
from sklearn.neighbors import NearestNeighbors

from joblib import Parallel, delayed

from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm

EPS = 1e-12  # used only for q construction; never added to transition denominators

# Colab 7-core setting
N_JOBS = 7

# Exact same half-moon design as the uploaded shared-readout notebook
N_PER_MOON = 150
K_NN = 15
SEED = 0
OUTPUT_DIR = "ollivier_results"

# Paper-convention gamma values.  These reproduce the old code's
# 0, 1.5, 2.5 conductance strengths.
GAMMAS = (0.0, 3.0, 5.0)

# Primary figure: jitter has actual cross-moon bridges but no clutter.
# Set RUN_CLUTTER = True to also generate the clutter diagnostic.
RUN_CLUTTER = False

SETTINGS = {
    "jitter": {"noise": 0.18, "n_clutter": 0},
    "clutter": {"noise": 0.08, "n_clutter": 100},
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# %%
# ============================================================
# Same data generation and base graph as the half-moon experiment
# ============================================================

def make_half_moons(
    n_per_moon=150,
    noise=0.0,
    n_clutter=0,
    seed=0,
    clutter_pad=0.5,
):
    rng = np.random.default_rng(seed)

    X, y = make_moons(
        n_samples=2 * n_per_moon,
        noise=noise,
        random_state=int(rng.integers(1 << 31)),
    )

    if n_clutter > 0:
        lower = X.min(axis=0) - clutter_pad
        upper = X.max(axis=0) + clutter_pad

        clutter = rng.uniform(
            lower,
            upper,
            size=(int(n_clutter), 2),
        )

        X = np.vstack([X, clutter])
        y = np.concatenate(
            [y, np.full(int(n_clutter), -1, dtype=int)]
        )

    return X, y


def gaussian_knn_kernel(X, k=15, bandwidth=None):
    n = X.shape[0]

    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    distances, indices = nn.kneighbors(X)

    if bandwidth is None:
        bandwidth = float(np.median(distances[:, 1:]))

    rows = np.repeat(np.arange(n), k)
    cols = indices[:, 1:].ravel()

    weights = np.exp(
        -(distances[:, 1:].ravel() ** 2)
        / (2.0 * bandwidth ** 2)
    )

    K = csr_matrix(
        (weights, (rows, cols)),
        shape=(n, n),
    )

    # Same symmetrization as the original notebook.
    K = K.maximum(K.T)

    return K, bandwidth


def gamma_dirichlet_conductance(K, gamma):
    """Paper convention:
    c_ij^(gamma) = q_i^(gamma/2) q_j^(gamma/2) K_ij.
    """
    degree = np.asarray(K.sum(axis=1)).ravel()
    q = (degree + EPS) / np.sum(degree + EPS)

    a = q ** (0.5 * gamma)
    A = diags(a)

    C = A @ K @ A
    return C.tocsr(), q


def escort_transition(K, gamma):
    """Numerically stable escort transition.

    Starting from
        c_ij^(gamma) = q_i^(gamma/2) q_j^(gamma/2) K_ij,
    the row factor q_i^(gamma/2) cancels exactly.  We therefore compute
        P_ij^(gamma) = q_j^(gamma/2) K_ij / sum_l q_l^(gamma/2) K_il
    directly, without adding a fixed epsilon to the gated degree.
    """
    C, q = gamma_dirichlet_conductance(K, gamma)

    a = q ** (0.5 * gamma)
    B = K @ diags(a)
    denom = np.asarray(B.sum(axis=1)).ravel()

    if np.any(denom <= 0.0):
        raise ValueError("Encountered a zero escort-transition denominator.")

    P = diags(1.0 / denom) @ B
    P = P.tocsr()

    row_sums = np.asarray(P.sum(axis=1)).ravel()
    err = float(np.max(np.abs(row_sums - 1.0)))
    if err > 1e-10:
        raise RuntimeError(f"Transition rows are not normalized: max error={err:.3e}")

    return P, q, C


def undirected_edges(K):
    upper = triu(K, k=1).tocoo()
    return upper.row.astype(int), upper.col.astype(int)

# %%
# ============================================================
# Exact Ollivier--Ricci curvature
# ============================================================

def fixed_ground_metric(X, K):
    upper = triu(K, k=1).tocoo()
    i = upper.row.astype(int)
    j = upper.col.astype(int)

    length = np.linalg.norm(
        X[i] - X[j],
        axis=1,
    )

    rows = np.concatenate([i, j])
    cols = np.concatenate([j, i])
    vals = np.concatenate([length, length])

    A = csr_matrix(
        (vals, (rows, cols)),
        shape=K.shape,
    )

    rho = shortest_path(
        A,
        directed=False,
        unweighted=False,
    )

    if not np.all(np.isfinite(rho)):
        raise ValueError(
            "The base graph is disconnected. "
            "Increase K_NN for this curvature calculation."
        )

    return rho


@lru_cache(maxsize=None)
def transport_constraint_matrix(n_source, n_target):
    """Marginal constraints for a transportation LP.

    The last target equation is omitted because total source and target
    masses are both one, so that equation is redundant.
    """
    Aeq = np.zeros(
        (
            n_source + n_target - 1,
            n_source * n_target,
        ),
        dtype=float,
    )

    for i in range(n_source):
        Aeq[
            i,
            i * n_target:(i + 1) * n_target,
        ] = 1.0

    for j in range(n_target - 1):
        Aeq[
            n_source + j,
            j::n_target,
        ] = 1.0

    return Aeq


def exact_wasserstein_1(a, b, cost):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    a = a / a.sum()
    b = b / b.sum()

    ns = len(a)
    nt = len(b)

    Aeq = transport_constraint_matrix(ns, nt)
    beq = np.concatenate([a, b[:-1]])

    result = linprog(
        np.asarray(cost, dtype=float).ravel(),
        A_eq=Aeq,
        b_eq=beq,
        bounds=(0.0, None),
        method="highs",
        options={"presolve": True},
    )

    if not result.success:
        raise RuntimeError(
            "Wasserstein LP failed: "
            + result.message
        )

    return float(result.fun)


def lazy_support(P, i, idleness=0.5):
    start = P.indptr[i]
    end = P.indptr[i + 1]

    neighbors = P.indices[start:end]
    masses = (
        (1.0 - idleness)
        * P.data[start:end]
    )

    # Combine the lazy atom with a possible self-transition.
    mass_dict = {
        int(k): float(v)
        for k, v in zip(neighbors, masses)
    }

    mass_dict[int(i)] = (
        mass_dict.get(int(i), 0.0)
        + float(idleness)
    )

    support = np.fromiter(
        mass_dict.keys(),
        dtype=int,
    )

    mass = np.fromiter(
        mass_dict.values(),
        dtype=float,
    )

    mass = mass / mass.sum()

    return support, mass


def edge_ollivier_curvature(
    i,
    j,
    P,
    rho,
    idleness=0.5,
):
    si, ai = lazy_support(
        P,
        int(i),
        idleness=idleness,
    )

    sj, aj = lazy_support(
        P,
        int(j),
        idleness=idleness,
    )

    cost = rho[np.ix_(si, sj)]

    W1 = exact_wasserstein_1(
        ai,
        aj,
        cost,
    )

    dij = float(
        rho[int(i), int(j)]
    )

    return (
        np.nan
        if dij <= 0
        else 1.0 - W1 / dij
    )


def classify_edges(y, edge_i, edge_j):
    out = np.empty(
        len(edge_i),
        dtype=object,
    )

    for r, (i, j) in enumerate(
        zip(edge_i, edge_j)
    ):
        if y[i] < 0 or y[j] < 0:
            out[r] = "clutter-involved"
        elif y[i] != y[j]:
            out[r] = "cross-moon"
        else:
            out[r] = "within-moon"

    return out


def compute_curvature(
    X,
    y,
    K,
    rho,
    gamma,
    n_jobs=7,
    idleness=0.5,
):
    P, q, C = escort_transition(
        K,
        gamma,
    )

    row_sums = np.asarray(P.sum(axis=1)).ravel()
    gated_degree = np.asarray(C.sum(axis=1)).ravel()
    print(
        f"gamma={gamma:g}: transition row-sum range "
        f"[{row_sums.min():.12f}, {row_sums.max():.12f}], "
        f"median gated degree={np.median(gated_degree):.3e}"
    )

    edge_i, edge_j = undirected_edges(K)

    print(
        f"gamma={gamma:g}: "
        f"{len(edge_i)} edges"
    )

    kappa = Parallel(
        n_jobs=n_jobs,
        prefer="threads",
        batch_size=16,
    )(
        delayed(edge_ollivier_curvature)(
            int(i),
            int(j),
            P,
            rho,
            idleness,
        )
        for i, j in zip(edge_i, edge_j)
    )

    kappa = np.asarray(
        kappa,
        dtype=float,
    )

    edge_type = classify_edges(
        y,
        edge_i,
        edge_j,
    )

    # Conductance on the same base edges.
    C_upper = triu(
        C,
        k=1,
    ).tocoo()

    c_lookup = {
        (int(i), int(j)): float(v)
        for i, j, v
        in zip(
            C_upper.row,
            C_upper.col,
            C_upper.data,
        )
    }

    c_gamma = np.array([
        c_lookup[(int(i), int(j))]
        for i, j
        in zip(edge_i, edge_j)
    ])

    return pd.DataFrame({
        "gamma": float(gamma),
        "i": edge_i,
        "j": edge_j,
        "y_i": y[edge_i],
        "y_j": y[edge_j],
        "edge_type": edge_type,
        "q_i": q[edge_i],
        "q_j": q[edge_j],
        "conductance_gamma": c_gamma,
        "ground_distance": rho[
            edge_i,
            edge_j,
        ],
        "kappa": kappa,
        "negative": kappa < 0,
    })

# %%
def four_node_check(
    eps_values=(1.0, 0.5, 0.2, 0.05),
):
    out = []

    for eps in eps_values:
        C = np.array([
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, eps, 0.0],
            [0.0, eps, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ])

        d = C.sum(axis=1)
        P = csr_matrix(
            C / d[:, None]
        )

        A = csr_matrix(
            (C > 0).astype(float)
        )

        rho = shortest_path(
            A,
            directed=False,
            unweighted=True,
        )

        numeric = edge_ollivier_curvature(
            1,
            2,
            P,
            rho,
            idleness=0.5,
        )

        theory = (
            -(1.0 - eps)
            / (1.0 + eps)
        )

        out.append({
            "epsilon": eps,
            "numeric": numeric,
            "theory": theory,
            "abs_error": abs(
                numeric - theory
            ),
        })

    check = pd.DataFrame(out)

    assert (
        check["abs_error"].max()
        < 1e-9
    )

    return check


four_node_check()

# %%
settings_to_run = ["jitter"]
if RUN_CLUTTER:
    settings_to_run.append("clutter")

objects = {}
edgewise_parts = []

for setting_name in settings_to_run:
    spec = SETTINGS[setting_name]

    print(
        "\nSETTING:",
        setting_name,
    )

    X, y = make_half_moons(
        n_per_moon=N_PER_MOON,
        noise=spec["noise"],
        n_clutter=spec["n_clutter"],
        seed=SEED,
    )

    K, bandwidth = gaussian_knn_kernel(
        X,
        k=K_NN,
    )

    rho = fixed_ground_metric(
        X,
        K,
    )

    edge_i, edge_j = (
        undirected_edges(K)
    )

    types = classify_edges(
        y,
        edge_i,
        edge_j,
    )

    print(
        "n =", len(y),
        "| edges =", len(edge_i),
        "| bandwidth =",
        round(bandwidth, 6),
    )

    print(
        "edge types =",
        pd.Series(types)
        .value_counts()
        .to_dict(),
    )

    setting_frames = []

    for gamma in GAMMAS:
        df = compute_curvature(
            X=X,
            y=y,
            K=K,
            rho=rho,
            gamma=gamma,
            n_jobs=N_JOBS,
            idleness=0.5,
        )

        df.insert(
            0,
            "setting",
            setting_name,
        )

        df.insert(
            1,
            "seed",
            SEED,
        )

        setting_frames.append(df)
        edgewise_parts.append(df)

    objects[setting_name] = {
        "X": X,
        "y": y,
        "K": K,
        "rho": rho,
        "bandwidth": bandwidth,
        "edgewise": pd.concat(
            setting_frames,
            ignore_index=True,
        ),
    }


edgewise = pd.concat(
    edgewise_parts,
    ignore_index=True,
)


def q25(x):
    return np.quantile(x, 0.25)


def q75(x):
    return np.quantile(x, 0.75)


summary = (
    edgewise
    .groupby(
        ["setting", "gamma", "edge_type"],
        as_index=False,
    )
    .agg(
        n_edges=("kappa", "size"),
        mean_kappa=("kappa", "mean"),
        sd_kappa=("kappa", "std"),
        median_kappa=("kappa", "median"),
        q25_kappa=("kappa", q25),
        q75_kappa=("kappa", q75),
        negative_fraction=("negative", "mean"),
        mean_conductance=(
            "conductance_gamma",
            "mean",
        ),
    )
)


edgewise_csv = os.path.join(
    OUTPUT_DIR,
    "half_moons_ollivier_edgewise.csv",
)

summary_csv = os.path.join(
    OUTPUT_DIR,
    "half_moons_ollivier_summary.csv",
)

edgewise.to_csv(
    edgewise_csv,
    index=False,
)

summary.to_csv(
    summary_csv,
    index=False,
)


print("\nSUMMARY")
print(
    summary[
        [
            "setting",
            "gamma",
            "edge_type",
            "n_edges",
            "median_kappa",
            "negative_fraction",
        ]
    ].to_string(index=False)
)

print("\nSaved:")
print(edgewise_csv)
print(summary_csv)


# Within-minus-cross median curvature contrast for the jitter diagnostic.
if "jitter" in settings_to_run:
    s0 = summary[summary["setting"] == "jitter"]
    pivot = s0.pivot(index="gamma", columns="edge_type", values="median_kappa")
    if {"within-moon", "cross-moon"}.issubset(pivot.columns):
        contrast = pivot["within-moon"] - pivot["cross-moon"]
        print("\nMedian curvature contrast (within - cross):")
        for g, v in contrast.items():
            print(f"  gamma={g:g}: {v:.6f}")

# %%
def plot_edge_map(
    X,
    y,
    K,
    edgewise_setting,
    setting_name,
):
    edge_i, edge_j = (
        undirected_edges(K)
    )

    segments = np.stack(
        [
            X[edge_i],
            X[edge_j],
        ],
        axis=1,
    )

    all_kappa = (
        edgewise_setting["kappa"]
        .to_numpy()
    )

    vmax = float(
        np.quantile(
            np.abs(all_kappa),
            0.995,
        )
    )

    vmax = max(
        vmax,
        0.25,
    )

    norm = TwoSlopeNorm(
        vmin=-vmax,
        vcenter=0.0,
        vmax=vmax,
    )

    fig, axes = plt.subplots(
        1,
        len(GAMMAS),
        figsize=(
            5.15 * len(GAMMAS),
            4.65,
        ),
        constrained_layout=True,
    )

    if len(GAMMAS) == 1:
        axes = [axes]

    last_collection = None

    for ax, gamma in zip(
        axes,
        GAMMAS,
    ):
        sub = (
            edgewise_setting[
                edgewise_setting[
                    "gamma"
                ] == gamma
            ]
            .sort_values(
                ["i", "j"]
            )
            .reset_index(drop=True)
        )

        kappa = (
            sub["kappa"]
            .to_numpy()
        )

        last_collection = LineCollection(
            segments,
            cmap="coolwarm",
            norm=norm,
            linewidths=0.85,
            alpha=0.80,
        )

        last_collection.set_array(
            kappa
        )

        ax.add_collection(
            last_collection
        )

        moon = y >= 0
        clutter = y < 0

        ax.scatter(
            X[moon, 0],
            X[moon, 1],
            c=y[moon],
            cmap="tab10",
            s=9,
            alpha=0.78,
            linewidths=0,
            zorder=3,
        )

        if np.any(clutter):
            ax.scatter(
                X[clutter, 0],
                X[clutter, 1],
                s=8,
                marker=".",
                alpha=0.45,
                zorder=3,
            )

        cross = (
            sub["edge_type"]
            .to_numpy()
            == "cross-moon"
        )

        if np.any(cross):
            midpoint = 0.5 * (
                X[edge_i[cross]]
                + X[edge_j[cross]]
            )

            ax.scatter(
                midpoint[:, 0],
                midpoint[:, 1],
                marker="x",
                s=30,
                linewidths=1.0,
                zorder=5,
            )

        within_kappa = sub.loc[
            sub["edge_type"]
            == "within-moon",
            "kappa",
        ]

        cross_kappa = sub.loc[
            sub["edge_type"]
            == "cross-moon",
            "kappa",
        ]

        title = (
            rf"$\gamma={gamma:g}$"
            + "\n"
            + "median within "
            + f"{within_kappa.median():.3f}"
        )

        if len(cross_kappa) > 0:
            title += (
                "\nmedian cross "
                + f"{cross_kappa.median():.3f}"
            )

        ax.set_title(title)
        ax.set_aspect(
            "equal",
            adjustable="datalim",
        )
        ax.autoscale()
        ax.axis("off")

    fig.colorbar(
        last_collection,
        ax=axes,
        shrink=0.88,
        pad=0.02,
        label=(
            r"Ollivier--Ricci curvature "
            r"$\kappa_{\gamma,1/2}$"
        ),
    )

    fig.suptitle(
        "Escort-walk Ollivier--Ricci curvature: "
        + setting_name,
        fontsize=13,
    )

    png = os.path.join(
        OUTPUT_DIR,
        "half_moons_ollivier_edges_"
        + setting_name
        + ".png",
    )

    pdf = os.path.join(
        OUTPUT_DIR,
        "half_moons_ollivier_edges_"
        + setting_name
        + ".pdf",
    )

    fig.savefig(
        png,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)

    return png, pdf


edge_map_paths = {}

for setting_name in settings_to_run:
    obj = objects[setting_name]

    edge_map_paths[
        setting_name
    ] = plot_edge_map(
        obj["X"],
        obj["y"],
        obj["K"],
        obj["edgewise"],
        setting_name,
    )


print("\nSaved edge maps:")
for key, value in edge_map_paths.items():
    print(key, "->", value)

# %%
def plot_by_type(
    summary,
    setting_name,
):
    sub = summary[
        summary["setting"]
        == setting_name
    ].copy()

    fig, ax = plt.subplots(
        figsize=(6.2, 4.4)
    )

    for edge_type in (
        "within-moon",
        "cross-moon",
        "clutter-involved",
    ):
        s = (
            sub[
                sub["edge_type"]
                == edge_type
            ]
            .sort_values("gamma")
        )

        if s.empty:
            continue

        g = s["gamma"].to_numpy()
        med = (
            s["median_kappa"]
            .to_numpy()
        )
        lo = (
            s["q25_kappa"]
            .to_numpy()
        )
        hi = (
            s["q75_kappa"]
            .to_numpy()
        )

        ax.plot(
            g,
            med,
            marker="o",
            label=edge_type,
        )

        ax.fill_between(
            g,
            lo,
            hi,
            alpha=0.16,
        )

    ax.axhline(
        0.0,
        linewidth=1.0,
    )

    ax.set_xlabel(
        r"$\gamma$"
    )

    ax.set_ylabel(
        r"$\kappa_{\gamma,1/2}$"
    )

    ax.set_title(
        "Curvature by edge type: "
        + setting_name
    )

    ax.legend(
        frameon=False
    )

    png = os.path.join(
        OUTPUT_DIR,
        "half_moons_ollivier_by_type_"
        + setting_name
        + ".png",
    )

    pdf = os.path.join(
        OUTPUT_DIR,
        "half_moons_ollivier_by_type_"
        + setting_name
        + ".pdf",
    )

    fig.savefig(
        png,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)

    return png, pdf


by_type_paths = {}

for setting_name in settings_to_run:
    by_type_paths[
        setting_name
    ] = plot_by_type(
        summary,
        setting_name,
    )


print("\nSaved summary figures:")
for key, value in by_type_paths.items():
    print(key, "->", value)
