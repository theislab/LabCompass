import numpy as np
import torch

from sc_exp_design.constants import DataFields
from sc_exp_design.networks.blocks import BaseModule, BaseForwardModel
from sc_exp_design.config.velocity_field import NeuralVelocityFieldConfig
from sc_exp_design.models.inverse_utils import LangevinOptimizer
from sc_exp_design.models.flow_matching import FlowMatching
from sc_exp_design.models.inverse import InverseModel
from sc_exp_design.sym import get_annotated_perturbation_data
from sc_exp_design.utils import set_reproducibility

# defining batch size
batch_size = 64
n_iters = 10
n_samples = 3

seed = 42


# running the simulation
sym_conf = {
    "sigma": 1.0,
    "d": 2,
    "U": 5,
    "n_cat": 4,
    "N0": 300,
    "Nu": 300,
    "mean_range": 5,
    "linespace_width": 30,
    "uniform_range": 10,
    "seed": 44,
    "return_perturbation_representation": True,
}

train_adata, sym_dict = get_annotated_perturbation_data(
    **sym_conf
)

class TestInverseUtils:
    
    @staticmethod
    def forward_pass(
        X_controls: torch.Tensor,
        cond: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
        forward_model: BaseForwardModel,
        target_predictor: BaseModule,
        n_samples: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """"""
        # Expand target  and controls
        target = {
            covariate: covariate_data.repeat(n_samples, X_controls.shape[0], 1).to(X_controls.device) for covariate, covariate_data in target.items()
        }
        X_controls = X_controls.unsqueeze(0).expand(n_samples, -1, -1) 

        # prepare batch information cellFlow           
        batch_dict = {
            DataFields.SOURCE_STATE: X_controls,
        }
        expanded_perturbation_data = {}
        for pert_key in cond:
            expanded_perturbation_data[pert_key] = cond[pert_key].unsqueeze(1).expand(-1, X_controls.shape[1], -1)
        batch_dict[DataFields.PERTURBATION_DATA] = expanded_perturbation_data

        # pushing forward the particles 
        X_pert_pred = forward_model.predict(
            batch_dict,
            no_grad=False,
        )

        class_pred = target_predictor(X_pert_pred)
        return class_pred["cell_type"], target["cell_type"]

    @staticmethod
    def manual_langevin_step(
        cond: dict[str, torch.Tensor],
        loss: torch.Tensor,
        noise_scale: float,
        eta: float,
        sqrt_eta: float,
        noise=None
    ) -> dict[str, torch.Tensor]:
        """"""
        out = {}        
        for pert, pert_data in cond.items():
            grad = torch.autograd.grad(loss, pert_data, create_graph=False, retain_graph=True)[0]
            with torch.no_grad():  # Fix: Avoid unnecessary detaching/reseting requires_grad
                if noise is None:
                    noise = torch.randn_like(pert_data) * noise_scale
                out[pert] = pert_data - (eta / 2) * grad + sqrt_eta * noise
        return out

    def prepare_forward_model(
        self,
    ) -> None:
        """"""
        # initializing flow matching model
        self.flow_matching = FlowMatching()

        # prepare data
        self.flow_matching.prepare_train_data(
            train_adata,
            control_key="control",
            perturbations=("treatment", "cell_type", ),
            perturbation_reps={"treatment": "treatment_shift"},
            use_perturbation_target_repr=True,
            perturbation_target_covariates={"cell_type": "label"},
        )

        # configure velocity field
        config = NeuralVelocityFieldConfig(
            sym_conf["d"], # dimensionality of the flow
            encode_state=False,
            perturbation_layers_before_pooling={
                "repr_treatment_treatment_shift": {
                    "input_dim": sym_conf["d"],
                }
            },
            decoder_mlp_kwargs={
                "hidden_dims": (64, 64, 64),
                "use_batchnorm": False,
                "use_dropout": False,
                "activation_class": torch.nn.ELU,
                "final_activation_class": torch.nn.Identity,
            }
        )

        # preparing the model
        self.flow_matching.prepare_model(
            config,
            optimizer_class=torch.optim.Adam,
            optimizer_kwargs={"lr":1e-4},
            num_time_steps=2,
        )

        # traininhg the flow
        self.flow_matching.train(
            num_training_steps=n_iters,
            train_batch_size=batch_size,
        )

    def prepare_inverse_model(
        self,
    ) -> None:
        """"""
        # initializing inverse model
        self.langevin_inverse_model = InverseModel(
            forward_model=self.flow_matching,
            inverse_method="langevin",
        )

        # preparing predictor for target covariates
        self.langevin_inverse_model.prepare_target_prediction_model(
            ("cell_type", ),
            {"cell_type": sym_conf["n_cat"]},
            target_covariates_noise_models={"cell_type": None},
            target_covariates_predictor_kwargs={
                "cell_type": {
                    "hidden_dims": (),
                    "use_batchnorm": False,
                    "use_dropout": False,
                    "dropout_rate": 0.0,
                    "activation_class": torch.nn.Identity,
                    "final_activation_class": torch.nn.Identity,
                }
            },
            optimizer_class=torch.optim.Adam,
            optimizer_kwargs={"lr": 1e-2},
        )

        # training the target prediction model
        self.langevin_inverse_model.train_target_prediction_model(
            num_training_steps=n_iters,
            train_batch_size=batch_size,
        )

        self.langevin_inverse_model.prepare_inverse_model(
            torch.tensor([0.0, 0.5, 0.0, 0.5]),
            torch.nn.functional.cross_entropy,
            "repr_treatment_treatment_shift",
            sym_conf["d"],
            {"repr_treatment_treatment_shift": False},
            n_samples=n_samples,
        )

    def test_langevin_optimizer_noise_free(
        self,
    ) -> None:
        """"""
        # prepare for test
        self.prepare_forward_model()
        self.prepare_inverse_model()

        # retrieving control states
        x_controls = torch.from_numpy(train_adata[train_adata.obs.control == True].X).cuda()

        cond_init = self.langevin_inverse_model.inverse_model.optimized_perturbation_data
        cond_init_clone0 = {cov:init.clone().detach().requires_grad_() for cov, init in cond_init.items()}
        cond_init_clone1 = {cov:init.clone().detach().requires_grad_() for cov, init in cond_init.items()}

        # check that we are starting from the same values
        assert torch.equal(cond_init['repr_treatment_treatment_shift'], cond_init_clone0['repr_treatment_treatment_shift'])
        assert torch.equal(cond_init['repr_treatment_treatment_shift'], cond_init_clone1['repr_treatment_treatment_shift'])

        # defining store for clones
        clones0_store = [cond_init_clone0['repr_treatment_treatment_shift'].clone().detach()]
        clones1_store = [cond_init_clone1['repr_treatment_treatment_shift'].clone().detach()]

        # defining store for losses
        clones0_loss_store = [torch.tensor(np.inf)]
        clones1_loss_store = [torch.tensor(np.inf)]

        print("starting optimization at: ", cond_init['repr_treatment_treatment_shift'])

        # attaching the optimizer to the first clone
        optim = LangevinOptimizer(list(cond_init_clone0.values()), noise_scale=0.00)

        for step in range(n_iters):

            cond_init_clone1 = {cov:init.clone().detach().requires_grad_() for cov, init in cond_init_clone1.items()}
            
            # sampling batch
            batch_idxs = np.random.choice(np.arange(x_controls.shape[0]), batch_size)
            x_controls_batch = x_controls[batch_idxs]

            # forward pass with first clone to get predicted and target classes
            pred_clone0, target_clone0 = self.forward_pass(
                x_controls_batch,
                cond_init_clone0,
                self.langevin_inverse_model.optimal_condition,
                self.flow_matching,
                self.langevin_inverse_model.target_prediction_model,
                self.langevin_inverse_model.inverse_model.n_samples,
            )

            # forward pass with first clone to get predicted and target classes
            pred_clone1, target_clone1 = self.forward_pass(
                x_controls_batch,
                cond_init_clone1,
                self.langevin_inverse_model.optimal_condition,
                self.flow_matching,
                self.langevin_inverse_model.target_prediction_model,
                self.langevin_inverse_model.inverse_model.n_samples,
            )

            # computing loss with the two clones
            loss_clone0 = torch.nn.functional.cross_entropy(pred_clone0, target_clone0)
            loss_clone1 = torch.nn.functional.cross_entropy(pred_clone1, target_clone1)

            # backpropagating on first clone with optimizer
            optim.zero_grad()
            loss_clone0.backward()
            optim.step()

            # backpropagating on second clone manually
            cond_init_clone1 = self.manual_langevin_step(
                cond_init_clone1,
                loss_clone1,
                optim.param_groups[0]["noise_scale"],
                optim.param_groups[0]["eta"],
                optim.param_groups[0]["sqrt_eta"],
            )

            if optim.param_groups[0]["noise_scale"] == 0.0:
                # updated values should be equal
                assert torch.equal(cond_init_clone0["repr_treatment_treatment_shift"], cond_init_clone1["repr_treatment_treatment_shift"])

                # computed losses should be equal
                assert torch.equal(loss_clone0, loss_clone1)
            
            # we should have moved somewhere
            assert not torch.equal(cond_init["repr_treatment_treatment_shift"], cond_init_clone0["repr_treatment_treatment_shift"])
            assert not torch.equal(cond_init["repr_treatment_treatment_shift"], cond_init_clone1["repr_treatment_treatment_shift"])

    def test_langevin_optimizer(
        self,
    ) -> None:
        """"""
        # prepare for test
        self.prepare_forward_model()
        self.prepare_inverse_model()

        # retrieving control states
        x_controls = torch.from_numpy(train_adata[train_adata.obs.control == True].X).cuda()

        cond_init = self.langevin_inverse_model.inverse_model.optimized_perturbation_data
        cond_init_clone0 = {cov:init.clone().detach().requires_grad_() for cov, init in cond_init.items()}
        cond_init_clone1 = {cov:init.clone().detach().requires_grad_() for cov, init in cond_init.items()}

        # check that we are starting from the same values
        assert torch.equal(cond_init['repr_treatment_treatment_shift'], cond_init_clone0['repr_treatment_treatment_shift'])
        assert torch.equal(cond_init['repr_treatment_treatment_shift'], cond_init_clone1['repr_treatment_treatment_shift'])

        print("starting optimization at: ", cond_init['repr_treatment_treatment_shift'])

        # attaching the optimizer to the first clone
        optim = LangevinOptimizer(list(cond_init_clone0.values()), noise_scale=0.00)

        # defining the store for the results
        clone0_store = torch.zeros((n_iters, n_samples, sym_conf["d"]))
        clone0_loss_store = torch.zeros((n_iters))
        
        clone1_store = torch.zeros((n_iters, n_samples, sym_conf["d"]))
        clone1_loss_store = torch.zeros((n_iters))

        # setting reproducibility
        set_reproducibility(seed)

        for step in range(n_iters):
            
            # sampling batch
            batch_idxs = np.random.choice(np.arange(x_controls.shape[0]), batch_size)
            x_controls_batch = x_controls[batch_idxs]

            # forward pass with first clone to get predicted and target classes
            pred_clone0, target_clone0 = self.forward_pass(
                x_controls_batch,
                cond_init_clone0,
                self.langevin_inverse_model.optimal_condition,
                self.flow_matching,
                self.langevin_inverse_model.target_prediction_model,
                self.langevin_inverse_model.inverse_model.n_samples,
            )

            # computing loss with the two clones
            loss_clone0 = torch.nn.functional.cross_entropy(pred_clone0, target_clone0)

            # backpropagating on first clone with optimizer
            optim.zero_grad()
            loss_clone0.backward()
            optim.step()

            # we should have moved somewhere
            assert not torch.equal(cond_init["repr_treatment_treatment_shift"], cond_init_clone0["repr_treatment_treatment_shift"])

            # storing current step
            clone0_store[step] = cond_init_clone0["repr_treatment_treatment_shift"].detach().cpu()
            clone0_loss_store[step] = loss_clone0.item()

        # setting reproducibility
        set_reproducibility(seed)


        for step in range(n_iters):

            cond_init_clone1 = {cov:init.clone().detach().requires_grad_() for cov, init in cond_init_clone1.items()}

            # sampling batch
            batch_idxs = np.random.choice(np.arange(x_controls.shape[0]), batch_size)
            x_controls_batch = x_controls[batch_idxs]

            # forward pass with first clone to get predicted and target classes
            pred_clone1, target_clone1 = self.forward_pass(
                x_controls_batch,
                cond_init_clone1,
                self.langevin_inverse_model.optimal_condition,
                self.flow_matching,
                self.langevin_inverse_model.target_prediction_model,
                self.langevin_inverse_model.inverse_model.n_samples,
            )

            loss_clone1 = torch.nn.functional.cross_entropy(pred_clone1, target_clone1)

            # backpropagating on second clone manually
            cond_init_clone1 = self.manual_langevin_step(
                cond_init_clone1,
                loss_clone1,
                optim.param_groups[0]["noise_scale"],
                optim.param_groups[0]["eta"],
                optim.param_groups[0]["sqrt_eta"],
            )
            
            # we should have moved somewhere
            assert not torch.equal(cond_init["repr_treatment_treatment_shift"], cond_init_clone1["repr_treatment_treatment_shift"])

            # storing current step
            clone1_store[step] = cond_init_clone1["repr_treatment_treatment_shift"].detach().cpu()
            clone1_loss_store[step] = loss_clone1.item()

        # check that optimization dynamics is the samecl
        assert torch.allclose(clone0_store, clone1_store), f"{clone0_store=}\n{clone1_store}"
        assert torch.allclose(clone0_loss_store, clone1_loss_store), f"{clone0_loss_store=}\n{clone1_loss_store}"
