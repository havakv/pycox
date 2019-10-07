import pytest
import numpy as np
import torch
from pycox.models.utils import pad_col, make_subgrid, cumsum_reverse

@pytest.mark.parametrize('val', [0, 1, 5])
def test_pad_col_start(val):
    x = torch.ones((2, 3))
    x_pad = pad_col(x, val, where='start')
    pad = torch.ones(2, 1) * val
    assert (x_pad == torch.cat([pad, x], dim=1)).all()

@pytest.mark.parametrize('val', [0, 1, 5])
def test_pad_col_end(val):
    x = torch.ones((2, 3))
    x_pad = pad_col(x, val)
    pad = torch.ones(2, 1) * val
    assert (x_pad == torch.cat([x, pad], dim=1)).all()

@pytest.mark.parametrize('n', [2, 13, 40])
def test_make_subgrid_1(n):
    grid = np.random.uniform(0, 100, n)
    grid = np.sort(grid)
    new_grid = make_subgrid(grid, 1)
    assert len(new_grid) == len(grid)
    assert (new_grid == grid).all()

@pytest.mark.parametrize('sub', [2, 10, 20])
@pytest.mark.parametrize('start', [0, 2])
@pytest.mark.parametrize('stop', [4, 100])
@pytest.mark.parametrize('n', [5, 10])
def test_make_subgrid(sub, start, stop, n):
    grid = np.linspace(start, stop, n)
    new_grid = make_subgrid(grid, sub)
    true_new = np.linspace(start, stop, n*sub - (sub-1))
    assert len(new_grid) == len(true_new)
    assert np.abs(true_new - new_grid).max() < 1e-13

def test_cumsum_reverse_error_dim():
    x = torch.randn((5, 3))
    with pytest.raises(NotImplementedError):
        cumsum_reverse(x, dim=0)
    with pytest.raises(NotImplementedError):
        cumsum_reverse(x, dim=2)

def test_cumsum_reverse_dim_1():
    torch.manual_seed(1234)
    x = torch.randn(5, 16)
    res_np = x.numpy()[:, ::-1].cumsum(1)[:, ::-1]
    res = cumsum_reverse(x, dim=1)
    assert np.abs(res.numpy() - res_np).max() < 1e-6
