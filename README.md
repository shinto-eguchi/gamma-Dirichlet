````markdown
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
├── half_moons_section51.ipynb
├── half_moons_section51.py
├── swiss_roll_section52.ipynb
├── swiss_roll_section52.py
├── figure_snippets.tex
├── requirements.txt
├── environment.yml
├── CITATION.cff
├── LICENSE
├── .gitignore
└── README.md
````

Generated figures and tables are saved under `gdc_figures/` and `results/`
when the scripts are executed.

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
jupyter notebook half_moons_section51.ipynb
jupyter notebook swiss_roll_section52.ipynb
```

## Running the scripts

```bash
python half_moons_section51.py
python swiss_roll_section52.py
```

The Swiss-roll experiment uses `N_REP = 8` and `N_JOBS = 7` in the notebook/script;
adjust these constants near the top of the file if needed.

## Notes

* In the half-moon experiment, scatter clutter is excluded from ARI scoring and
  only affects the graph construction.
* In the Swiss-roll experiment, clutter points affect the graph and embedding,
  while the geodesic-preservation score is computed only on the original manifold
  points.
* The Swiss-roll input figure uses a truncated viridis colormap for the clean
  manifold and magenta points for uniform clutter.

## Citation

If you use this code, please cite the associated manuscript. A preliminary
`CITATION.cff` file is included and can be updated once the final bibliographic
information is available.

```
```
