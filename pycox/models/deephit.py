"""Impleentation of the deephit model, but currently for single event cause
(not compteeting risk).
"""
import torch
from pyth import Model, tuplefy

def nll_pmf(phi, y, d, reduction='elementwise_mean', _epsilon=1e-7):
    """Negative log-likelihood for survival data from probability mass function (pmf).
    
    Arguments:
        phi {torch.tensor} -- Predictions. phi.softmax() gives the pmf estimate.
        y {torch.tensor} -- Labels (0/1), with 1 at the event/censoring time.
        d {torch.tensor} -- Indicator 1: event, 0: censoring.
    
    Keyword Arguments:
        reduction {str} -- Reductions (default: {'elementwise_mean'})
        _epsilon {float} -- For numerical stability. (default: {1e-7})
    
    Returns:
        torch.tensor -- The nll.
    """
    # This is a more numerical stable version of:
    # 
    # d.view(-1)
    # f = phi.softmax(1)
    # s = 1. - f.cumsum(1)
    # part1 = f.mul(y).sum(1).log().mul(d)
    # part2 = s.mul(y).sum(1).log().mul(1. - d)
    # loss =  - part1.add(part2)
    # return loss.mean()
    d.view(-1)
    gamma = phi.max(1)[0]
    cumsum = phi.sub(gamma.view(-1, 1)).exp().cumsum(1)
    sum_ = cumsum[:, -1]
    part1 = phi.mul(y).sum(1).sub(gamma).mul(d)
    part2 = - sum_.log()
    # part 3 should be replaced by a reverse sum as it is more stable
    part3 = sum_.sub(cumsum.mul(y).sum(1)).add(_epsilon).log().mul(1. - d)
    loss = - part1.add(part2).add(part3)
    if reduction == 'none':
        return loss
    elif reduction == 'elementwise_mean':
        return loss.mean()
    return loss.sum()


def loss_deephit_l2(phi, y, d, sigma=1., reduction='elementwise_mean', _epsilon=1e-7):
    idx_sort = y.nonzero()[:, 1].sort()[1]
    phi, y, d = phi[idx_sort], y[idx_sort], d[idx_sort]
    cif = phi.softmax(1).cumsum(1)
    # f_si = phi.mul(y).sum(1)
    loss = torch.zeros_like(d)
    _acc = torch.zeros_like(d)
    _etas = torch.zeros_like(d)
    for row, idx in y[:-1].nonzero():
#         acceptable = y[(row+1):, idx].eq(0.).float().mul(d[(row+1):]).mul(d[row])
        acceptable = y[(row+1):, idx].mul(d[(row+1):]).eq(0.).float().mul(d[row])
        if acceptable.sum() == 0:
            loss[row] = 0.
            continue
        eta = (-cif[row, idx].sub(cif[(row+1):, idx]).div(sigma)).exp()
#         print(eta)
        loss[row] = eta.mul(acceptable).sum() / acceptable.sum()
        _acc[row] = acceptable.sum()
        _etas[row] = eta.mul(acceptable).sum()
#     print('end')
    if reduction == 'none':
        return loss#, _acc, _etas
    elif reduction == 'elementwise_mean':
#         return loss.mean()
        n = loss.nonzero().shape[0]
        loss = loss.sum()
        if n != 0:
            loss = loss / n
        return loss
    return loss.sum()
