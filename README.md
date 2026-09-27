# LabCompass

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

## Package structure

- `labcompass.models` — the user-facing model classes listed above.
- `labcompass.networks` — the underlying neural network architectures: velocity fields, flow-map
  networks, noise/likelihood models, attention and condition-encoding building blocks.
- `labcompass.data` — data loading and management for `AnnData`-based perturbation datasets
  (control/treatment sampling, batching, schema validation).
- `labcompass.couplings` — optimal-transport and independent couplings pairing control and
  perturbed cells during training.
- `labcompass.flows` — the interpolation paths / noise schedules used by conditional flow matching
  (rectified, variance-preserving, ...).
- `labcompass.ode` — ODE integration utilities for pushing cell states forward through a trained
  velocity field.
- `labcompass.inverse` — training-free guided sampling from a pretrained flow: `LossGuidedFlow` steers
  generation towards a target loss, and `ImpliciDualGuidedFlow` handles inequality constraints via KKT conditions.
- `labcompass.training` — training loops and loss/metric callbacks for each model type.
- `labcompass.transforms` — invertible pre/post-processing transforms (standardization,
  composition of transforms).
- `labcompass.metrics` — evaluation metrics, both distributional (e.g. MMD, Wasserstein distance,
  energy distance) and classification-based.
- `labcompass.sym` — synthetic/toy data generators used for testing and examples.

## Example usage

A minimal end-to-end run on the synthetic data shipped with the package:

```python
import numpy as np
import torch

import labcompass

# synthetic AnnData with control and perturbed cells
adata = labcompass.sym.get_dummy_adata(states=np.random.normal(size=(850, 200)).astype(np.float32))

# initializing the model
cfm = labcompass.models.FlowMatching(coupling_type="ot", device_id="cpu")

# preparing the train data
cfm.prepare_train_data(
    adata,
    sample_rep="states",                                   # `.obsm` key holding the cell states
    control_key="is_control",                              # boolean `.obs` column flagging control cells
    perturbations=("treatment0",),                         # `.obs` columns holding the perturbations
    perturbation_reps={"treatment0": "treatment0_label"},  # representation of each perturbation
)

# configuring the conditional velocity field; each perturbation representation gets its own
# encoder, keyed as `repr_<perturbation>_<representation>`
config = labcompass.config.NeuralVelocityFieldConfig(
    flow_dim=adata.obsm["states"].shape[1],
    encode_conditions=True,
    perturbation_layers_before_pooling={
        "repr_treatment0_treatment0_label": {"input_dim": 5, "output_dim": 16},
    },
)

# preparing and training the model
cfm.prepare_model(config)
cfm.train(num_training_steps=500)

# predicting perturbed states from control cells
control_states = torch.from_numpy(adata.obsm["states"][adata.obs["is_control"].values][:16])
preds = cfm.predict(
    {
        "condition": {"repr_treatment0_treatment0_label": torch.eye(5)[[0] * 16]},
        "source": control_states,
    },
)
```

Validation sets can be registered with `cfm.prepare_validation_data(name, adata)` and are evaluated every
`valid_freq` steps during `cfm.train(...)`.

## Installation

You need to have Python 3.10 or newer installed on your system.
If you don't have Python installed, we recommend installing [Miniforge][].

Install the latest development version:

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

See [CHANGELOG.md](CHANGELOG.md).

## Contact

For questions, help requests or bug reports, please use the [issue tracker][].

## Citation

> t.b.a

[miniforge]: https://github.com/conda-forge/miniforge
[issue tracker]: https://github.com/theislab/LabCompass/issues
