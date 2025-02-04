# scExpDesign

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/lorenzo-consoli/scExpDesign/test.yaml?branch=main
[badge-docs]: https://img.shields.io/readthedocs/scExpDesign

Generative Modeling for Experimental Design in Single Cell Data

## Getting started

Please refer to the [documentation][],
in particular, the [API documentation][].

## Example usage

```{python}
>>> # importing the required packages
>>> import sc_exp_design
>>> import anndata as ad
>>> # initializing the AnnData object with the train data
>>> train_adata = ad.AnnData(...)
>>> # retrieving the default configurations
>>> config = sc_exp_design.networks.NeuralVelocityFieldConfig()
>>> # initializing the model with default settings
>>> cfm = sc_exp_design.models.FlowMatching()
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
...    config, # configurations for the conditioinal velocity field
... )
>>> # training the model
>>> cfm.train()
```

## Installation

You need to have Python 3.10 or newer installed on your system.
If you don't have Python installed, we recommend installing [Mambaforge][].

There are several alternative options to install scExpDesign:

<!--
1) Install the latest release of `scExpDesign` from [PyPI][]:

```bash
pip install scExpDesign
```
-->

1. Install the latest development version:

```bash
pip install git+https://github.com/lorenzo-consoli/scExpDesign.git@main
```

2. Run Tests

```bash
PYTHONUNBUFFERED=1 pytest --tb=long --capture=tee-sys tests/networks/test_velocity_field.py 2>&1 | tee .pytest-logs.log
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
[issue tracker]: https://github.com/lorenzo-consoli/scExpDesign/issues
[tests]: https://github.com/lorenzo-consoli/scExpDesign/actions/workflows/test.yml
[documentation]: https://scExpDesign.readthedocs.io
[changelog]: https://scExpDesign.readthedocs.io/en/latest/changelog.html
[api documentation]: https://scExpDesign.readthedocs.io/en/latest/api.html
[pypi]: https://pypi.org/project/scExpDesign
