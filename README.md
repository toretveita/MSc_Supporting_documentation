## Supporting documentation repository for "Business process mining in smart homes"

This repository contains the artifacts listed in Appendix A of in the thesis "Business process mining in smart homes". The repository is organized into three main folders:

- ML: containing machine-learning scripts and results
- inputs: containing input files (Raw and filtered for the two datasets used, CASAS and REFIT)
- prom: containing the results from proM tasks (discovery, conformance and hold-out). 

The following instructions are for running the scripts after cloning the repository. 


## Running the ML scripts

The machine-learning experiments live in the `ML/` folder:

- `ML/ML_casas.py` — CASAS: compares **raw vs PM-prepared** logs across model families.
- `ML/architectural_ablation_casas.py` — CASAS: architectural ablation study (PM-prepared log).
- `ML/ML_refit.py` — REFIT: model-family comparison.

All three scripts:

- Expect an XES event log (`.xes`) as input.
- Write JSON results to `ML/results/`.
- Cache intermediate fold results in `ML/results/checkpoints/` (so reruns are faster).

### 1) Create a Python environment

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 2) Install dependencies

Install all required Python packages:

```powershell
python -m pip install -r requirements.txt
```

At minimum, the ML scripts need:

- `numpy`, `pandas`
- `scikit-learn`
- `scipy`
- `torch`
- `torch-geometric` (PyG)


### 3) Provide your XES files (via CLI args)

This repository already contains the input logs under `inputs/`:

- CASAS: `inputs/casas/shib010.xes`
- REFIT: `inputs/refit/refit_building02.xes`

REFIT model-family comparison (defaults to the included REFIT log):

```powershell
python .\ML\ML_refit.py
```

CASAS architectural ablation (defaults to the included CASAS log):

```powershell
python .\ML\architectural_ablation_casas.py
```

CASAS raw-vs-prepared comparison requires **two** logs (raw and PM-prepared):

```powershell
python .\ML\ML_casas.py --raw-xes-path .\inputs\casas\shib010.xes --prepared-xes-path C:\path\to\your\pm_prepared_casas.xes
```

To see all options for a script:

```powershell
python .\ML\ML_refit.py --help
```

### 4) Run the experiments

Run each script from the repository root:

```powershell
python .\ML\ML_refit.py
python .\ML\architectural_ablation_casas.py
python .\ML\ML_casas.py
```

### Outputs

- Results JSON files are written to `ML/results/` (filenames are set inside each script).
- Checkpoints are written to `ML/results/checkpoints/`.
- To force a full retrain (ignore checkpoints), pass `--force-retrain`.
- To disable checkpoints entirely, pass `--no-use-checkpoints`.

For a quick test (fast, not the full experiment), reduce folds/epochs, e.g.:

```powershell
python .\ML\ML_refit.py --n-folds 2 --max-traces 10 --gnn-epochs 1 --lstm-epochs 1 --no-use-checkpoints --force-retrain
```

