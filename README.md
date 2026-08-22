# LabCompass

[![Documentation][badge-docs]][documentation]

[badge-docs]: https://img.shields.io/readthedocs/LabCompass

Generative modeling for experimental design in single-cell data.

LabCompass is built around **conditional flow matching (CFM)**: given control cells and a set of
perturbations (drugs, genetic edits, dosage/time covariates, ...), it learns a velocity field that
transports control cell states to their perturbed counterparts. On top of this core, LabCompass
provides a family of models for the design and analysis of perturbation experiments:

- **`FlowMatching`** — the core conditional flow matching model: learns a neural velocity field
  from control to perturbed cell states, conditioned on arbitrary perturbation covariates.
- **`FlowMatchingWithScore`** — extends `FlowMatching` with a learned score function alongside the
  velocity field.
- **`FlowMap`** — a flow-map variant of `FlowMatching` for direct (few-step) transport between
  states, using either independent or optimal-transport (OT) couplings.
- **`InverseModel`** — solves the inverse problem: given a desired target cell state, infers the
  perturbation covariates that would produce it (via MAP optimization, Langevin sampling, or an
  amortized neural inverse model), using a trained `FlowMatching` model as the forward model.
- **`TargetPredictionModel`** — a supervised model that predicts downstream target covariates from
  cell states, e.g. for use as a forward/proxy model in guided generation.

## Getting started

Please refer to the [documentation][],
in particular, the [API documentation][].

## Example usage

```{python}
>>> # importing the required packages
>>> import labcompass
>>> import anndata as ad
>>> # initializing the AnnData object with the train data
>>> train_adata = ad.AnnData(...)
>>> # retrieving the default configurations
>>> config = labcompass.networks.NeuralVelocityFieldConfig()
>>> # initializing the model with default settings
>>> cfm = labcompass.models.FlowMatching()
>>> # preparing the train data
>>> cfm.prepare_train_data(
...     train_adata,
...     control_key="is_control",
...     perturbations=("Drug1", "Drug2"),
...     perturbation_covariates={
...            "Drug1": ("time", "dosage"),
...            "Drug2": ("time", "dosage"),
...     },
...     perturbation_reps={
...            "Drug1": ("drug_id", ),
...            "Drug2": ("drug_id", ),
...    },
... )
>>> # preparing the model
>>> cfm.prepare_model(
...    2, # dimensionality of the flow
...    config, # configurations for the conditional velocity field
... )
>>> # training the model
>>> cfm.train()
```

## Installation

You need to have Python 3.10 or newer installed on your system.
If you don't have Python installed, we recommend installing [Mambaforge][].

There are several alternative options to install LabCompass:

<!--
1) Install the latest release of `LabCompass` from [PyPI][]:

```bash
pip install LabCompass
```
-->

1. Install the latest development version:

```bash
pip install git+https://github.com/theislab/LabCompass.git@main
```

## Development

To run the test suite locally:

```bash
pip install -e ".[test]"
PYTHONUNBUFFERED=1 pytest --tb=long --capture=tee-sys 2>&1 | tee .pytest-logs.log
```

## Release notes

See the [changelog][].

## Contact

For questions and help requests, you can reach out in the [scverse discourse][].
If you found a bug, please use the [issue tracker][].

## Citation

> t.b.a

[mambaforge]: https://github.com/conda-forge/miniforge#mambaforge
[scverse discourse]: https://discourse.scverse.org/
[issue tracker]: https://github.com/theislab/LabCompass/issues
[documentation]: https://LabCompass.readthedocs.io
[changelog]: https://LabCompass.readthedocs.io/en/latest/changelog.html
[api documentation]: https://LabCompass.readthedocs.io/en/latest/api.html
[pypi]: https://pypi.org/project/LabCompass
