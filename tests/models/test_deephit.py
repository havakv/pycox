import pytest
import torch
from pycox.models.deephit import (nll_pmf_cr, nll_pmf, pair_rank_mat, rank_loss_deephit_cr, 
    rank_loss_deephit_single, LossDeepHitSingle, LossDeepHit)

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
    r1 = nll_pmf_cr(phi, idx_duration, events)
    r2 = nll_pmf(phi.view(batch * n_risk, -1), idx_duration, events.float())
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
    r1 = rank_loss_deephit_cr(phi, idx_duration, events, rank_mat, sigma)
    r2 = rank_loss_deephit_single(phi.view(batch, -1), idx_duration, events.float(),
        rank_mat, sigma)
    assert (r1 - r2).abs() < 1e-6


@pytest.mark.parametrize('seed', [0, 1, 2])
@pytest.mark.parametrize('m', [1, 8])
@pytest.mark.parametrize('sigma', [0.1, 0.2, 1.])
@pytest.mark.parametrize('alpha', [1, 0.5, 0.1, 0.])
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
    loss_cr = LossDeepHit(alpha, sigma)
    loss_single = LossDeepHitSingle(alpha, sigma)
    r1 = loss_cr(phi, idx_duration, events, rank_mat)
    r2 = loss_single(phi.view(batch, -1), idx_duration, events.float(), rank_mat)
    assert (r1 - r2).abs() < 1e-5