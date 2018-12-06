"""Impleentation of the deephit model, but currently for single event cause
(not compteeting risk).
"""
from pyth import Model, tuplefy

def pmf_nll(phi, y, d, reduction='elementwise_mean', _epsilon=1e-7):
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