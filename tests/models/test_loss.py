import pytest
import numpy as np
import torch
import torchtuples as tt

from pycox.models.data import pair_rank_mat
from pycox.models import loss

@pytest.mark.parametrize('seed', [0, 1, 2])
@pytest.mark.parametrize('m', [1, 5, 8])
def test_nll_pmf_cr_equals_nll_pmf(seed, m):
    torch.manual_seed(seed)
    # m = 5
    n_risk = 1
    rep = 7
    batch = m * (n_risk + 1) * rep
    phi = torch.randn(batch, n_risk, m)
    idx_duration = torch.arange(m).repeat(rep * (n_risk + 1))
    events = torch.arange(n_risk + 1).repeat(m * rep)
    r1 = loss.nll_pmf_cr(phi, idx_duration, events)
    r2 = loss.nll_pmf(phi.view(batch * n_risk, -1), idx_duration, events.float())
    assert (r1 - r2).abs() < 1e-5

@pytest.mark.parametrize('seed', [0, 1, 2])
@pytest.mark.parametrize('m', [1, 5, 8])
@pytest.mark.parametrize('sigma', [0.1, 0.2, 1.])
def test_rank_loss_deephit_cr_equals_single(seed, m, sigma):
    torch.manual_seed(seed)
    n_risk = 1
    rep = 7
    batch = m * (n_risk + 1) * rep
    phi = torch.randn(batch, n_risk, m)
    idx_duration = torch.arange(m).repeat(rep * (n_risk + 1))
    events = torch.arange(n_risk + 1).repeat(m * rep)
    rank_mat = pair_rank_mat(idx_duration.numpy(), events.numpy())
    rank_mat = torch.tensor(rank_mat)
    r1 = loss.rank_loss_deephit_cr(phi, idx_duration, events, rank_mat, sigma)
    r2 = loss.rank_loss_deephit_single(phi.view(batch, -1), idx_duration, events.float(),
        rank_mat, sigma)
    assert (r1 - r2).abs() < 1e-6


@pytest.mark.parametrize('seed', [0, 1])
@pytest.mark.parametrize('m', [1, 8])
@pytest.mark.parametrize('sigma', [0.1, 1.])
@pytest.mark.parametrize('alpha', [1, 0.5, 0.])
def test_loss_deephit_cr_equals_single(seed, m, sigma, alpha):
    torch.manual_seed(seed)
    n_risk = 1
    rep = 7
    batch = m * (n_risk + 1) * rep
    phi = torch.randn(batch, n_risk, m)
    idx_duration = torch.arange(m).repeat(rep * (n_risk + 1))
    events = torch.arange(n_risk + 1).repeat(m * rep)
    rank_mat = pair_rank_mat(idx_duration.numpy(), events.numpy())
    rank_mat = torch.tensor(rank_mat)
    loss_cr = loss.DeepHitLoss(alpha, sigma)
    loss_single = loss.DeepHitSingleLoss(alpha, sigma)
    r1 = loss_cr(phi, idx_duration, events, rank_mat)
    r2 = loss_single(phi.view(batch, -1), idx_duration, events.float(), rank_mat)
    assert (r1 - r2).abs() < 1e-5

@pytest.mark.parametrize('seed', [0, 1])
@pytest.mark.parametrize('shrink', [0, 0.01, 1.])
def test_cox_cc_loss_single_ctrl(seed, shrink):
    np.random.seed(seed)
    n = 100
    case = np.random.uniform(-1, 1, n)
    ctrl = np.random.uniform(-1, 1, n)
    case, ctrl = tt.tuplefy(case, ctrl).to_tensor()
    loss_1 = loss.cox_cc_loss(case, (ctrl,), shrink)
    loss_2 = loss.cox_cc_loss_single_ctrl(case, ctrl, shrink)
    assert (loss_1 - loss_2).abs() < 1e-6

@pytest.mark.parametrize('shrink', [0, 0.01, 1.])
def test_cox_cc_loss_single_ctrl_zero(shrink):
    n = 10
    case = ctrl = torch.zeros(n)
    loss_1 = loss.cox_cc_loss_single_ctrl(case, ctrl, shrink)
    val = torch.tensor(2.).log()
    assert (loss_1 - val).abs() == 0

@pytest.mark.parametrize('shrink', [0, 0.01, 1.])
def test_cox_cc_loss_zero(shrink):
    n = 10
    case = ctrl = torch.zeros(n)
    loss_1 = loss.cox_cc_loss(case, (ctrl,), shrink)
    val = torch.tensor(2.).log()
    assert (loss_1 - val).abs() == 0
