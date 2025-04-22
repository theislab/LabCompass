import pytest

import anndata
import numpy as np


@pytest.fixture
def adata() -> anndata.AnnData:
    """"""

    # defining state data
    num_unique_treatments = 5
    num_unique_groups = 3
    num_control_cells = 100
    num_perturbed_cells = 150
    tot_perturbed_cells = num_perturbed_cells*num_unique_treatments
    num_cells = num_control_cells + tot_perturbed_cells
    num_genes = 200
    num_perturbation_feats = 100
    states = np.ones((num_cells, num_genes))

    # defining treatment data
    treatment0_label = "treatment0"
    treatment1_label = "treatment1"
    treatment2_label = "treatment2" # to be put in anndata.obsm
    control_key = "is_control"

    # defining label maps
    id_to_label_map = {
        0: "control",
        **{
            u: f"drug{u}" for u in range(1, num_unique_treatments + 1)
        },
    }
    label_to_id_map = {
        v: k for k, v in id_to_label_map.items()
    }

    # defining group maps
    pert_to_group_map = {
        "control": 0,
        **{
            f"drug{u}": (u%num_unique_groups + 1) for u in range(1, num_unique_treatments + 1)
        }
    }

    # defining function for retrieving all the perturbation data associated to a treatment
    def get_treatment_data():        
        # (perturbation covariates) sampling dosages
        dosages = np.random.uniform(size=(tot_perturbed_cells, )) 
        # (perturbation covariates) sampling times
        times = np.random.uniform(size=(tot_perturbed_cells, ))
        # (perturbation reps) retrieving perturbation ids
        perturbation_ids =  np.concatenate([np.ones((num_perturbed_cells))*u for u in range(1, num_unique_treatments + 1)])
        # (perturbation reps) retrieving perturbation labels
        perturbation_labels = np.vectorize(id_to_label_map.get)(perturbation_ids)
        # (perturbation reps) retrieving perturbation group label
        group_labels = np.vectorize(pert_to_group_map.get)(perturbation_ids)
        # sampling perturbation features
        perturbation_features = np.random.randn(tot_perturbed_cells, num_perturbation_feats)
        return dosages, times, perturbation_labels, group_labels, perturbation_features

    # retrieving data for treatment0
    treatment0_dosages, treatment0_times, treatment0_perturbation_labels, treatment0_group_labels, _ = get_treatment_data()
    # retrieving data for treatment1
    treatment1_dosages, treatment1_times, treatment1_perturbation_labels, treatment1_group_labels, _ = get_treatment_data()
    # retrieving data for treatment2
    _, _, _, _, treatment2_features = get_treatment_data()

    # shuffling treatment0 data
    shuffled_indices = np.random.permutation(tot_perturbed_cells)
    treatment0_perturbation_labels = treatment0_perturbation_labels[shuffled_indices]
    shuffled_indices = np.random.permutation(tot_perturbed_cells)
    treatment0_group_labels = treatment0_group_labels[shuffled_indices]

    # shuffling treatment1 data
    shuffled_indices = np.random.permutation(tot_perturbed_cells)
    treatment1_perturbation_labels = treatment1_perturbation_labels[shuffled_indices]
    shuffled_indices = np.random.permutation(tot_perturbed_cells)
    treatment1_group_labels = treatment1_group_labels[shuffled_indices]

    # defining perturbation data for control states
    control_dosages = np.zeros((num_control_cells, )) 
    control_times = np.zeros((num_control_cells, ))
    control_pertubation_labels = np.array(["control" for _ in range(num_control_cells)])
    control_perturbation_features = np.zeros((num_control_cells, num_perturbation_feats))
    control_group_labels = np.array(["control" for _ in range(num_control_cells)])

    # concatenating control perturbation data with treatment0 data
    treatment0_dosages = np.concatenate((control_dosages, treatment0_dosages), axis=0)
    treatment0_times = np.concatenate((control_times, treatment0_times), axis=0)
    treatment0_perturbation_labels = np.concatenate((control_pertubation_labels, treatment0_perturbation_labels), axis=0)
    treatment0_group_labels = np.concatenate((control_group_labels, treatment0_group_labels), axis=0)

    # concatenating control perturbation data with treatment0 data
    treatment1_dosages = np.concatenate((control_dosages, treatment1_dosages), axis=0)
    treatment1_times = np.concatenate((control_times, treatment1_times), axis=0)
    treatment1_perturbation_labels = np.concatenate((control_pertubation_labels, treatment1_perturbation_labels), axis=0)
    treatment1_group_labels = np.concatenate((control_group_labels, treatment1_group_labels), axis=0)

    # concatenating control perturbation data with treatement2 data
    treatment2_features = np.concatenate((control_perturbation_features, treatment2_features), axis=0)

    # defining flat for control cells
    is_control = np.concatenate(
        (
            np.array([True for _ in range(num_control_cells)]),
            np.array([False for _ in range(tot_perturbed_cells)]),
        ), axis=0
    )

    # target covariates
    target0_num_unique_values = 7
    target0_label = "target0"
    target1_num_unique_values = 9
    target1_label = "target1"

    # defining function for retrieving all target covariate data
    def get_target_data(num_unique_values):
        target_id_to_label_map = {
            idx:f"tgt{idx}" for idx in range(num_unique_values)
        }
        target_ids = np.random.randint(0, num_unique_values, (num_cells,))
        target_labels = np.vectorize(target_id_to_label_map.get)(target_ids)
        return target_labels

    # retrieving target data
    target0_data = get_target_data(target0_num_unique_values)
    target1_data = get_target_data(target1_num_unique_values)
    
    # shuffling target data
    shuffled_indices = np.random.permutation(num_cells)
    target0_data = target0_data[shuffled_indices]
    shuffled_indices = np.random.permutation(num_cells)
    target1_data = target1_data[shuffled_indices]

    # defining mappings for uns
    uns = {
        f"{treatment0_label}_label": {label: np.array([label_id]) for label, label_id in label_to_id_map.items()},
        f"{treatment0_label}_group": {label: np.array([group_id]) for label, group_id in pert_to_group_map.items()},
        f"{treatment1_label}_label": {label: np.array([label_id]) for label, label_id in label_to_id_map.items()},
        f"{treatment1_label}_group": {label: np.array([group_id]) for label, group_id in pert_to_group_map.items()},
    }

    # definig mappings for obs
    obs = {
        control_key: is_control,
        target0_label: target0_data,
        target1_label: target1_data,
        treatment0_label: treatment0_perturbation_labels,
        treatment1_label: treatment1_perturbation_labels,
    }

    # defining mappings obsm
    sample_rep = "states"
    obsm = {
        sample_rep: states,
        f"{treatment0_label}_dose": treatment0_dosages,
        f"{treatment0_label}_time": treatment0_times,
        f"{treatment1_label}_dose": treatment1_dosages,
        f"{treatment1_label}_time": treatment1_times,
        f"{treatment2_label}_features": treatment2_features,
    }

    return anndata.AnnData(
        X=states,
        obs=obs,
        uns=uns,
        obsm=obsm,
    )
