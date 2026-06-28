# Gamma-Dirichlet graph experiments for robust manifold learning

This repository contains the reproducibility code for the numerical experiments in
Section 5 of the manuscript on density-gated graph conductance and robust manifold
learning.

The code is organized around two experiments:

- **Section 5.1:** half-moons with scatter noise; Gamma-Dirichlet clustering is
  compared with Coifman--Lafon diffusion-map clustering on a shared Gaussian kNN graph.
- **Section 5.2:** Swiss roll with uniform off-manifold clutter; density removal and
  dimension gating are compared through geodesic-preservation correlations.

All code comments are written in English.

## Repository structure

```text
.
├── notebooks/
│   ├── half_moons_section51.ipynb
│   └── swiss_roll_section52.ipynb
├── scripts/
│   ├── half_moons_section51.py
│   └── swiss_roll_section52.py
├── tex/
│   └── figure_snippets.tex
├── gdc_figures/        # generated figures; ignored except .gitkeep
├── results/            # optional generated outputs; ignored except .gitkeep
├── requirements.txt
├── environment.yml
├── CITATION.cff
├── LICENSE
└── README.md
```

## Installation

Using `pip`:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Using `conda`:

```bash
conda env create -f environment.yml
conda activate gamma-dirichlet-manifold-experiments
```

## Running the notebooks

```bash
jupyter notebook notebooks/half_moons_section51.ipynb
jupyter notebook notebooks/swiss_roll_section52.ipynb
```

## Running the scripts

```bash
python scripts/half_moons_section51.py
python scripts/swiss_roll_section52.py
```

The scripts save figures and tables under `gdc_figures/` by default.  The
Swiss-roll experiment uses `N_REP = 8` and `N_JOBS = 7` in the notebook/script;
adjust these constants near the top of the file if needed.

## Notes

- In the half-moon experiment, scatter clutter is excluded from ARI scoring and
  only affects the graph construction.
- In the Swiss-roll experiment, clutter points affect the graph and embedding,
  while the geodesic-preservation score is computed only on the original manifold
  points.
- The Swiss-roll input figure uses a truncated viridis colormap for the clean
  manifold and magenta points for uniform clutter.

## Citation

If you use this code, please cite the associated manuscript.  A preliminary
`CITATION.cff` file is included and can be updated once the final bibliographic
information is available.
