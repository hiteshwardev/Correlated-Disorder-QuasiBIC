# Correlated disorder and the radiative loss of a photonic-crystal quasi-bound state in the continuum

Hitesh Kumar Singh

This repository contains the complete computational study behind the article of the same title. A single notebook, `notebook.ipynb`, builds the photonic-crystal structures, runs every guided-mode-expansion (GME) and rigorous coupled-wave (RCWA) calculation, writes all intermediate data to `results/`, draws the figures of the article into `figures/`, and closes with a table of the main results. `results/summary.json` collects the numbers quoted in the article, and the stage records in `results/` hold the per-case values behind its tables. The notebook also explains the physics, the mathematical formulation and the algorithms used at each stage.

## The study in brief

A symmetry-protected bound state in the continuum (BIC) at the Γ point of a photonic-crystal slab (structure A: square lattice of air holes, radius 0.18a, slab thickness 0.782a, ε = 12) is made lossy by Gaussian-correlated fluctuations of the hole radii in N × 1 supercells. At fixed disorder amplitude σ the ensemble-mean loss ⟨1/Q⟩ is compared across correlation lengths ℓc, with 680 disorder realisations in total.

* Correlations of one to two lattice constants raise the loss by a factor of about 1.7 relative to white disorder (1.69 [1.19, 2.39] at ℓc = a and 1.69 [1.21, 2.35] at ℓc = 2a, 95% intervals); at ℓc = 0.5a no change is detected (0.90 [0.67, 1.21]).
* The excess loss scales as σ^β with β = 1.95 ± 0.10, as expected at second order in the disorder.
* A spectral-overlap model predicts the enhancement without adjustable parameters and follows its growth with N, but its coupling constant drifts with N, so the model is phenomenological.
* Structure B (slab thickness 0.9225a) shows the signature of a merged BIC at a low plane-wave cutoff, but its Q scales as k^−2 at every cutoff, like an isolated BIC; its apparent Q advantage over structure A falls from about 10^4 to 1.45 as the cutoff grows and is an artefact of basis truncation.

## Repository layout

```
notebook.ipynb     the complete study: background, computations, analysis, figures, final numbers
src/               building blocks called from the notebook
  config.py        parameters of the study and the seed scheme
  geometry.py      photonic-crystal slab geometries (legume objects)
  gme.py           GME runs, mode tracking and the two loss estimators
  disorder.py      Gaussian-correlated disorder by circulant spectral synthesis
  ensembles.py     disorder ensembles with per-realisation checkpointing
  model.py         spectral-overlap model of the disorder-induced loss
  rcwa.py          RCWA reflectance and Fano fit for the benchmark grating
  stats.py         bootstrap intervals, random-effects pooling, power-law fits
  records.py       JSON records and on-disk caching
  plotting.py      figure style and automatic layout checks
results/           JSON records of every computed stage
figures/           the nine figures of the article, as PDF and 600 dpi PNG
requirements.txt   pinned Python dependencies
environment.yml    the same environment for conda
```

## Installation

Python 3.11 was used. With pip:

```
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

or with conda:

```
conda env create -f environment.yml
conda activate qbic-disorder
```

To work with the notebook interactively, also install JupyterLab (`pip install jupyterlab`).

## Running the study

Open `notebook.ipynb` and run all cells in order, or execute it from the command line:

```
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebook.ipynb
```

Every stage that takes more than a few seconds stores its result as a JSON record in `results/`. A stage whose record exists is loaded rather than recomputed, so with the records shipped here the notebook runs in a few minutes; it still recomputes three stored disorder realisations from their seeds and checks that they agree to machine precision. Deleting `results/`, or setting `RECOMPUTE = True` in the setup cell, regenerates everything. Single-core times of a full recomputation on the machine used for the study (about two hours in total):

| Stage | Notebook section | Time |
|---|---|---|
| Parameter survey, momentum-space diagnostic, benchmarks | 4, 5 | 6 min |
| Plane-wave cutoff ladder and guided-mode enrichment | 6 | 12 min |
| Main ensemble grid (12 points, 40 realisations each) | 8 | 52 min |
| Supercell-size series and cutoff check | 8 | 45 min |
| Analysis and figures | 9 to 11 | 1 min |

Ensemble stages write each finished realisation to a checkpoint file, so an interrupted run resumes where it stopped.

## Reproducibility

Every disorder realisation has its own integer seed, derived from the grid point and the replicate number (`src/config.py`), and the numerical libraries are restricted to one thread, so repeated runs on the same machine give identical results. The records in `results/` come from a full recomputation in a fresh virtual environment built from `requirements.txt`, started from an empty `results/` directory; running the notebook on these records regenerates every figure and `results/summary.json` unchanged. With other processors or linear-algebra libraries the last digits of the results can differ.

The notebook guards its own inputs. Before any calculation, Section 3.2 checks the building blocks: unique seeds, the normalisation of the disorder spectrum, exact stripe widths in the RCWA grid, the scaling of the model and known answers of the statistical routines. When the ensembles are loaded, Section 8.1 checks that every record holds exactly the seeds prescribed for its grid point and recomputes three stored realisations from their seeds.

## Figures

| File | Article figure |
|---|---|
| `fig01_supercell` | Fig. 1, disorder realisations of the supercell |
| `fig02_cutoff_ladder` | Fig. 2, convergence with the plane-wave cutoff |
| `fig03_correlation_length` | Fig. 3, mean loss against correlation length |
| `fig04_estimators` | Fig. 4, agreement of the two loss estimators |
| `fig05_distribution` | Fig. 5, distribution of the loss |
| `fig06_amplitude` | Fig. 6, scaling with the disorder amplitude |
| `fig07_supercell_size` | Fig. 7, enhancement at three supercell sizes |
| `fig08_momentum_space` | Fig. 8, clean-lattice Q(k) of structures A and B |
| `fig09_apparent_enhancement` | Fig. 9, apparent Q enhancement against the cutoff |

## Software

The GME calculations use [legume](https://github.com/fancompute/legume) (MIT License) and the RCWA benchmark uses [grcwa](https://github.com/weiliangjinca/grcwa) (GNU General Public License v3). Both are installed as dependencies and are not part of this repository.

## Citation

If you use this code or data, please cite the article: H. K. Singh, "Correlated disorder and the radiative loss of a photonic-crystal quasi-bound state in the continuum" (2026), together with this repository.

## License

The code and data in this repository are released under the MIT License (see `LICENSE`).
