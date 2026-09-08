# Dual-Porosity Model for Fractured Chalk

Python implementation of a dual-porosity model for variably saturated water flow and conservative solute transport in fractured chalk.

This repository contains the numerical model implementations used in the associated study, including a constant-parameter formulation and a saturation-dependent formulation in which the water-exchange coefficient and/or dispersivity vary with the modeled water state.

## Repository structure

```text
dual-porosity-chalk-model/
├── model/
│   ├── dual_porosity_constant.py
│   └── dual_porosity_saturation_dependent.py
├── outputs/
├── README.md
├── requirements.txt
└── .gitignore
```

## Model implementations

### Constant-parameter model

`model/dual_porosity_constant.py`

Runs the dual-porosity water-flow and conservative-solute-transport model using case-specific constant values of the water-exchange coefficient and dispersivity.

### Saturation-dependent model

`model/dual_porosity_saturation_dependent.py`

Runs an internal constant-parameter reference simulation first. The modeled reference-state evolution is then used to define the start and end anchors of the saturation-dependent parameter relationships.

The user can select one of three formulations:

- `omega_variable`
- `dispersivity_variable`
- `both_variable`

The numerical implementation retains the coupled mobile-immobile water and solute calculations used in the study.

## Requirements

The archived version was run with Python 3.13 and the following package versions:

```text
numpy==2.2.2
pandas==2.2.3
scipy==1.15.2
```

Install the required packages with:

```bash
python -m pip install -r requirements.txt
```

## Running the model

Choose the desired saturation case near the top of the model script:

```python
selected_case = "60"
```

Available cases are:

```text
"NS"
"60"
"40"
```

For the saturation-dependent model, also choose the parameter mode:

```python
comparison_mode = "both_variable"
```

Then run, for example:

```bash
python model/dual_porosity_constant.py
```

or:

```bash
python model/dual_porosity_saturation_dependent.py
```

## Numerical outputs

The scripts automatically write numerical results to the `outputs/` directory.

Outputs include model time series, breakthrough curves, cumulative water flux, water-content evolution, state/parameter evolution, final depth profiles, and run-summary information.

The saturation-dependent script saves results for both its internal constant reference simulation and the selected variable-parameter simulation.

## Main numerical dependencies

- **NumPy** — numerical array operations
- **pandas** — tabular output handling
- **SciPy** — nonlinear least-squares solution of the implicit water-flow system

## Reproducibility

The scripts in this repository are intended to preserve the numerical implementation used for the associated study. Plotting, observational-data comparison, and model-performance post-processing are kept separate from the core model implementation.

## Citation

A formal software citation will be provided through the archived Zenodo release.

If you use this code before the DOI is added, please cite the associated study and reference this repository.

## Version

The first archived release will be tagged:

`v1.0.0`
