import numpy as np
import torch
import pytest
from sklearn.metrics import r2_score

from sc_exp_design.metrics.metrics import (
    compute_r_squared,
    compute_e_distance,
    compute_mmd,
    compute_wasserstein_distance,
    compute_weighted_min_mse,
    compute_weighted_max_mse,
    compute_marginal_e_distance,
    compute_n_gen_knn_cell_props,
    compute_cell_props,
    compute_coclustering
)

# Sample data
@pytest.fixture(params=[
    (10, 5),       # basic: 10 samples, 5 features
    (1, 10, 5),    # batched: 1 batch, 10 samples, 5 features
    (50, 2),       # low-dim: 50 samples, 2 features
    (5, 50),       # fewer samples than features
])
def small_data(request):
    shape = request.param
    np.random.seed(0)
    torch.manual_seed(0)
    pred_np = np.random.randn(*shape)
    target_np = pred_np + np.random.normal(0, 0.1, size=pred_np.shape)
    weights_np = np.abs(np.random.randn(pred_np.shape[-1]))
    return torch.tensor(pred_np, dtype=torch.float32), torch.tensor(target_np, dtype=torch.float32), torch.tensor(weights_np, dtype=torch.float32)


def test_compute_r_squared(small_data):
    pred, target, weights = small_data
    if pred.ndim == 2:
        pred_np = pred.numpy()
        target_np = target.numpy()
        expected = r2_score(np.mean(pred_np, axis=0), np.mean(target_np, axis=0))
        result = compute_r_squared(pred, target)
        assert np.isclose(result, expected)
    else:
        result = compute_r_squared(pred, target)
        assert isinstance(result, float)

def test_compute_e_distance(small_data):
    pred, target, weights = small_data
    # Ensure batch dimension exists
    if pred.ndim == 2:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    result = compute_e_distance(pred, target)
    assert isinstance(result, float)
    assert result >= 0

def test_compute_mmd(small_data):
    pred, target, weights = small_data
    if pred.ndim == 2:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    val = compute_mmd(pred, target)
    assert isinstance(val, float)
    assert val >= 0

## add checking weights
def test_weighted_min_max_mse(small_data):
    pred, target, weights = small_data
    pred_np = pred.numpy()
    target_np = target.numpy()
    weights_np = weights.numpy()
    if pred_np.ndim == 3:
        for i in range(pred_np.shape[0]):
            min_mse = compute_weighted_min_mse(pred_np[i], target_np[i])
            max_mse = compute_weighted_max_mse(pred_np[i], target_np[i])
            min_weight_mse = compute_weighted_min_mse(pred_np[i], target_np[i], weights_np)
            max_weight_mse = compute_weighted_max_mse(pred_np[i], target_np[i], weights_np)
            assert isinstance(min_mse, np.float32)
            assert isinstance(max_mse, np.float32)
            assert isinstance(min_weight_mse, np.float32)
            assert isinstance(max_weight_mse, np.float32)
            assert min_mse <= max_mse
            assert min_weight_mse <= max_weight_mse
    else:
        min_mse = compute_weighted_min_mse(pred_np, target_np)
        max_mse = compute_weighted_max_mse(pred_np, target_np)
        min_weight_mse = compute_weighted_min_mse(pred_np, target_np, weights_np)
        max_weight_mse = compute_weighted_max_mse(pred_np, target_np, weights_np)
        assert isinstance(min_mse, np.float32)
        assert isinstance(max_mse, np.float32)
        assert isinstance(min_weight_mse, np.float32)
        assert isinstance(max_weight_mse, np.float32)
        assert min_mse <= max_mse
        assert min_weight_mse <= max_weight_mse

def test_compute_wasserstein_distance_exact(small_data):
    pred, target, _ = small_data
    val = compute_wasserstein_distance(pred, target, method="exact")
    assert isinstance(val, np.float32) or isinstance(val, np.float64)
    assert val >= 0

## add checking weights
def test_marginal_e_distance(small_data):
    pred, target, weights = small_data
    dist = compute_marginal_e_distance(pred, target, weights)
    assert isinstance(dist, np.float32) or isinstance(dist, np.float64)
    assert dist >= 0

def test_n_gen_knn_cell_props(small_data):
    pred, target, _ = small_data
    val = compute_n_gen_knn_cell_props(pred, target)
    assert 0 <= val <= 1

def test_cell_props(small_data):
    pred, target, _ = small_data
    val = compute_cell_props(pred, target, n_iter=5)  # Keep test fast
    assert 0 <= val <= 1

def test_compute_coclustering(small_data):
    pred, target, weights = small_data
    pred = pred
    target = target
    val = compute_coclustering(pred, target, k=3, resolution=0.5)
    assert isinstance(val, np.float32) or isinstance(val, np.float64)
    assert 0 <= val <= 1