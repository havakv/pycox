import torch
from pycox.models.utils import pad_col


def _reduction(loss, reduction='mean'):
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    raise ValueError(f"`reduction` = {reduction} is not valid. Use 'none', 'mean' or 'sum'.")


def nll_pmf(phi, idx_durations, events, reduction='mean', epsilon=1e-7):
    """Negative log-likelihood of the hazard parametrization model.
    See make_y for a better understanding of labeling scheeme.
    
    Arguments:
        phi {torch.tensor} -- Estimates in (-inf, inf), where hazard = sigmoid(phi).
        idx_durations {torch.tensor} -- Event times represented as indexes.
        events {torch.tensor} -- Indicator of event (1.) or censoring (0.).
            Same lenght as 'idx_durations'.
        reduction {string} -- How to reduce the loss.
            'none': No reduction.
            'mean': Mean of tensor.
            'sum: sum.
    
    Returns:
        torch.tensor -- Retruns mean negative log-likelihood.
    """
    if (idx_durations.max()) >= phi.shape[1]:
        raise ValueError("""'t_idx' too large. Probably need to increase output size of net.""")
    events = events.view(-1)
    idx_durations = idx_durations.view(-1, 1)
    phi = pad_col(phi)
    gamma = phi.max(1)[0]
    cumsum = phi.sub(gamma.view(-1, 1)).exp().cumsum(1)
    sum_ = cumsum[:, -1]
    part1 = phi.gather(1, idx_durations).view(-1).sub(gamma).mul(events)
    part2 = - sum_.relu().add(epsilon).log()
    part3 = sum_.sub(cumsum.gather(1, idx_durations).view(-1)).relu().add(epsilon).log().mul(1. - events)
    # need relu() in part3 (and possibly part2) because cumsum on gpu has some bugs and we risk getting negative numbers.
    loss = - part1.add(part2).add(part3)
    return _reduction(loss, reduction)


class _Loss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction


class NLLPMFLoss(_Loss):
    def forward(self, phi, idx_durations, events):
        return nll_pmf(phi, idx_durations, events, self.reduction)



