## Repository structure

.
├── half_moons_recalculation_shared_readout.ipynb
├── swiss_roll_section52.ipynb
├── swiss_roll_section52.py
├── figure_snippets.tex
├── requirements.txt
├── environment.yml
├── CITATION.cff
├── LICENSE
├── .gitignore
└── README.md
## Ollivier--Ricci curvature experiment

The edgewise Ollivier--Ricci curvature diagnostic for the half-moon experiment
is reproduced by

- `half_moons_ollivier_curvature.ipynb` — Colab/Jupyter notebook
- `half_moons_ollivier_curvature.py` — standalone Python script

The experiment uses 300 half-moon observations with Gaussian coordinate noise
of standard deviation 0.18 and no additional background clutter.  A Gaussian
15-nearest-neighbour graph is constructed, and the escort walk is evaluated at


```math
c_{ij}^{(\gamma)}
=
q_i^{\gamma/2} q_j^{\gamma/2} K_{ij}.
```

For the paper convention, the gamma-gated conductance is

$$c_{ij}^{(\gamma)}=q_i^{\gamma/2} q_j^{\gamma/2} K_{ij}.$$

The Ollivier--Ricci curvature is computed from the one-half lazy escort walk
using a fixed ground metric.  As gamma increases, within-moon edges become
more positively curved while cross-moon edges become more negatively curved.

For the realization used in the manuscript, the median curvatures are:

| gamma | within-moon | cross-moon | curvature contrast |
|---:|---:|---:|---:|
| 0 | 0.117 | -0.012 | 0.130 |
| 3 | 0.172 | -0.028 | 0.200 |
| 5 | 0.193 | -0.041 | 0.233 |

The standalone script uses 7 parallel workers by default.
