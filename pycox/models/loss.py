import torch
import torch.nn.functional as F
from pycox.models.utils import pad_col, log_softplus


def _reduction(loss, reduction='mean'):
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    raise ValueError(f"`reduction` = {reduction} is not valid. Use 'none', 'mean' or 'sum'.")

def nll_logistic_hazard(phi, idx_durations, events, reduction='mean'):
    """Negative log-likelihood of the hazard parametrization model.
    
    Arguments:
        phi {torch.tensor} -- Estimates in (-inf, inf), where hazard = sigmoid(phi).
        idx_durations {torch.tensor} -- Event times represented as indices.
        events {torch.tensor} -- Indicator of event (1.) or censoring (0.).
            Same lenght as 'idx_durations'.
        reduction {string} -- How to reduce the loss.
            'none': No reduction.
            'mean': Mean of tensor.
            'sum: sum.
    
    Returns:
        torch.tensor -- The negative log-likelihood.
    """
    events = events.view(-1, 1)
    idx_durations = idx_durations.view(-1, 1)
    y_bce = torch.zeros_like(phi).scatter(1, idx_durations, events)
    bce = F.binary_cross_entropy_with_logits(phi, y_bce, reduction='none')
    loss = bce.cumsum(1).gather(1, idx_durations).view(-1)
    return _reduction(loss, reduction)

def nll_pmf(phi, idx_durations, events, reduction='mean', epsilon=1e-7):
    """Negative log-likelihood for the pmf parametrized model.
    
    Arguments:
        phi {torch.tensor} -- Estimates in (-inf, inf), where pmf = somefunc(phi).
        idx_durations {torch.tensor} -- Event times represented as indices.
        events {torch.tensor} -- Indicator of event (1.) or censoring (0.).
            Same lenght as 'idx_durations'.
        reduction {string} -- How to reduce the loss.
            'none': No reduction.
            'mean': Mean of tensor.
            'sum: sum.
    
    Returns:
        torch.tensor -- The negative log-likelihood.
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

def nll_pc_hazard_loss(phi, idx_durations, events, interval_frac, reduction='mean'):
    """Negative log-likelihood of the PC-Hazard parametrization model.
    
    Arguments:
        phi {torch.tensor} -- Estimates in (-inf, inf), where hazard = sigmoid(phi).
        idx_durations {torch.tensor} -- Event times represented as indices.
        events {torch.tensor} -- Indicator of event (1.) or censoring (0.).
            Same lenght as 'idx_durations'.
        interval_frac {torch.tensor} -- Fraction of last interval before event/censoring.
        reduction {string} -- How to reduce the loss.
            'none': No reduction.
            'mean': Mean of tensor.
            'sum: sum.
    
    Returns:
        torch.tensor -- The negative log-likelihood.
    """
    idx_durations = idx_durations.view(-1, 1)
    events = events.view(-1)
    interval_frac = interval_frac.view(-1)

    keep = idx_durations.view(-1) >= 0
    phi = phi[keep, :]
    idx_durations = idx_durations[keep, :]
    events = events[keep]
    interval_frac = interval_frac[keep]

    # log_h_e = F.softplus(phi.gather(1, idx_durations).view(-1)).log().mul(events)
    log_h_e = log_softplus(phi.gather(1, idx_durations).view(-1)).mul(events)
    haz = F.softplus(phi)
    scaled_h_e = haz.gather(1, idx_durations).view(-1).mul(interval_frac)
    haz = pad_col(haz, where='start')
    sum_haz = haz.cumsum(1).gather(1, idx_durations).view(-1) 
    loss = - log_h_e.sub(scaled_h_e).sub(sum_haz)
    return _reduction(loss, reduction)


def _rank_loss_deephit(pmf, y, rank_mat, sigma, reduction='mean'):
    """Ranking loss from deephit.
    
    Arguments:
        pmf {torch.tensor} -- Matrix with probability mass function pmf_ij = f_i(t_j)
        y {torch.tensor} -- Matrix with indicator of duration and censoring time. 
        rank_mat {torch.tensor} -- See pair_rank_mat function.
        sigma {float} -- Sigma from DeepHit paper, choosen by you.
    
    Returns:
        torch.tensor -- loss
    """
    r = _diff_cdf_at_time_i(pmf, y)
    loss = rank_mat * torch.exp(-r/sigma)
    loss = loss.mean(1, keepdim=True)
    return _reduction(loss, reduction)

def _diff_cdf_at_time_i(pmf, y):
    """R is the matrix from the deephit code giving the difference in cdf between individual
    i and j, at the event time of j. 
    I.e: R_ij = F_i(T_i) - F_j(T_i)
    
    Arguments:
        pmf {torch.tensor} -- Matrix with probability mass function pmf_ij = f_i(t_j)
        y {torch.tensor} -- Matrix with indicator of duratio/censor time.

        # not_ec {torch.tensor} -- Indicator matrix with same shape as 'pmf', indicating
        #     that the individual has still not had an event or censoring.
    
    Returns:
        torch.tensor -- R_ij = F_i(T_i) - F_j(T_i)
    """
    n = pmf.shape[0]
    ones = torch.ones((n, 1), device=pmf.device)
    r = pmf.cumsum(1).matmul(y.transpose(0, 1))
    diag_r = r.diag().view(1, -1)
    r = ones.matmul(diag_r) - r
    return r.transpose(0, 1)

def rank_loss_deephit_single(phi, idx_durations, events, rank_mat, sigma, reduction='mean'):
    """Rank loss proposed by DeepHit authors for competing risks.
    
    Arguments:
        pmf {torch.tensor} -- Matrix with probability mass function pmf_ij = f_i(t_j)
        y {torch.tensor} -- Matrix with indicator of duration and censoring time. 
        rank_mat {torch.tensor} -- See pair_rank_mat function.
        sigma {float} -- Sigma from DeepHit paper, choosen by you.
    Arguments:
        phi {torch.tensor} -- Preditions as float tensor with shape [batch, n_durations]
            all in (-inf, inf).
        idx_durations {torch.tensor} -- Int tensor with index of durations.
        events {torch.tensor} -- Float indicator of event or censoring (1 is event).
        rank_mat {torch.tensor} -- See pair_rank_mat function.
        sigma {float} -- Sigma from DeepHit paper, choosen by you.
    
    Keyword Arguments:
        reduction {string} -- How to reduce the loss.
            'none': No reduction.
            'mean': Mean of tensor.
            'sum': sum.
    
    Returns:
        torch.tensor -- Rank loss.
    """
    idx_durations = idx_durations.view(-1, 1)
    events = events.float().view(-1)
    pmf = pad_col(phi).softmax(1)
    y = torch.zeros_like(pmf).scatter(1, idx_durations, 1.) # one-hot
    rank_loss = _rank_loss_deephit(pmf, y, rank_mat, sigma, reduction)
    return rank_loss

def nll_pmf_cr(phi, idx_durations, events, reduction='mean', epsilon=1e-7):
    """Negtive log-likelihood for pmf paraterizations. `phi` is the ''logit''.
    
    Arguments:
        phi {torch.tensor} -- Preditions as float tensor with shape [batch, n_risks, n_durations]
            all in (-inf, inf).
        idx_durations {torch.tensor} -- Int tensor with index of durations.
        events {torch.tensor} -- Int tensor with event types.
            {0: Censored, 1: first group, ..., n_risks: n'th risk group}.
    
    Keyword Arguments:
        reduction {string} -- How to reduce the loss.
            'none': No reduction.
            'mean': Mean of tensor.
            else: sum.
    
    Returns:
        torch.tensor -- Negative log-likelihood.
    """
    # Should improve numerical stbility by, e.g., log-sum-exp tric.
    events = events.view(-1) - 1
    event_01 = (events != -1).float()
    idx_durations = idx_durations.view(-1)
    batch_size = phi.size(0)
    sm = pad_col(phi.view(batch_size, -1)).softmax(1)[:, :-1].view(phi.shape)
    index = torch.arange(batch_size)
    part1 = sm[index, events, idx_durations].relu().add(epsilon).log().mul(event_01)
    part2 = (1 - sm.cumsum(2)[index, :, idx_durations].sum(1)).relu().add(epsilon).log().mul(1 - event_01)
    loss = - part1.add(part2)
    return _reduction(loss, reduction)

def rank_loss_deephit_cr(phi, idx_durations, events, rank_mat, sigma, reduction='mean'):
    """Rank loss proposed by DeepHit authors for competing risks.
    
    Arguments:
        phi {torch.tensor} -- Preditions as float tensor with shape [batch, n_risks, n_durations]
            all in (-inf, inf).
        idx_durations {torch.tensor} -- Int tensor with index of durations.
        events {torch.tensor} -- Int tensor with event types.
            {0: Censored, 1: first group, ..., n_risks: n'th risk group}.
        rank_mat {torch.tensor} -- See pair_rank_mat function.
        sigma {float} -- Sigma from DeepHit paper, choosen by you.
    
    Keyword Arguments:
        reduction {string} -- How to reduce the loss.
            'none': No reduction.
            'mean': Mean of tensor.
            else: sum.
    
    Returns:
        torch.tensor -- Rank loss.
    """
    idx_durations = idx_durations.view(-1)
    events = events.view(-1) - 1
    event_01 = (events == -1).float()

    batch_size, n_risks = phi.shape[:2]
    pmf = pad_col(phi.view(batch_size, -1)).softmax(1)
    pmf = pmf[:, :-1].view(phi.shape)
    y = torch.zeros_like(pmf)
    y[torch.arange(batch_size), :, idx_durations] = 1.

    loss = []
    for i in range(n_risks):
        rank_loss_i = _rank_loss_deephit(pmf[:, i, :], y[:, i, :], rank_mat, sigma, 'none')
        loss.append(rank_loss_i.view(-1) * (events == i).float())

    if reduction == 'none':
        return sum(loss)
    elif reduction == 'mean':
        return sum([lo.mean() for lo in loss])
    elif reduction == 'sum':
        return sum([lo.sum() for lo in loss])
    return _reduction(loss, reduction)


def cox_cc_loss(g_case, g_control, shrink=0., clamp=(-3e+38, 80.)):
    """Torch loss functin for the Cox case-control models.
    TODO:
        For only one control we should instead use the softplus function.
    
    Arguments:
        g_case {torch.Tensor} -- Resulat of net(input_case)
        g_control {torch.Tensor} -- Results of [net(input_ctrl1), net(input_ctrl2), ...]
    
    Keyword Arguments:
        shrink {float} -- Shinkage that encourage the net got give g_case and g_control
            closer to zero (a regularizer in a sense). (default: {0.})
        clamp {tuple} -- See code (default: {(-3e+38, 80.)})
    
    Returns:
        [type] -- [description]
    """
    control_sum = 0.
    shrink_control = 0.
    for ctr in g_control:
        shrink_control += ctr.abs().mean()
        ctr = ctr - g_case
        ctr = torch.clamp(ctr, *clamp)  # Kills grads for very bad cases (should instead cap grads!!!).
        control_sum += torch.exp(ctr)
    loss = torch.log(1. + control_sum)
    shrink_zero = shrink * (g_case.abs().mean() + shrink_control) / len(g_control)
    return torch.mean(loss) + shrink_zero.abs()

def cox_ph_loss(log_h, event, eps=1e-7):
    """Requires the input to be sorted by descending duration time.
    See DatasetDurationSorted.

    We calculate the negative log of $(\frac{h_i}{\sum_{j \in R_i} h_j})^d$,
    where h = exp(log_h) are the hazards and R is the risk set, and d is event.

    We just compute a cumulative sum, and not the true Risk sets. This is a
    limitiation, but simple and fast.
    """
    event = event.view(-1)
    log_h = log_h.view(-1)
    gamma = log_h.max()
    log_cumsum_h = log_h.sub(gamma).exp().cumsum(0).add(eps).log().add(gamma)
    return - log_h.sub(log_cumsum_h).mul(event).sum().div(event.sum())


class _Loss(torch.nn.Module):
    """Generic loss function.
    
    Arguments:
        reduction {string} -- How to reduce the loss.
            'none': No reduction.
            'mean': Mean of tensor.
            'sum: sum.
    """
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction


class NLLLogistiHazardLoss(_Loss):
    """Negative log-likelihood of the hazard parametrization model.
    See `loss.nll_logistic_hazard` for details.
    
    Arguments:
        reduction {string} -- How to reduce the loss.
            'none': No reduction.
            'mean': Mean of tensor.
            'sum: sum.
    
    Returns:
        torch.tensor -- The negative log-likelihood.
    """
    def forward(self, phi, idx_durations, events):
        return nll_logistic_hazard(phi, idx_durations, events, self.reduction)


class NLLPMFLoss(_Loss):
    """Negative log-likelihood of the pmf parametrization model.
    See `loss.nll_pmf` for details.
    
    Arguments:
        reduction {string} -- How to reduce the loss.
            'none': No reduction.
            'mean': Mean of tensor.
            'sum: sum.
    
    Returns:
        torch.tensor -- The negative log-likelihood.
    """
    def forward(self, phi, idx_durations, events):
        return nll_pmf(phi, idx_durations, events, self.reduction)


class NLLPCHazardLoss(_Loss):
    def forward(self, phi, idx_durations, events, interval_frac, reduction='mean'):
        """Negative log-likelihood of the PC-Hazard parametrization model.
        See `loss.nll_pc_hazard_loss` for details.
    
        Arguments:
            reduction {string} -- How to reduce the loss.
                'none': No reduction.
                'mean': Mean of tensor.
                'sum: sum.
    
        Returns:
            torch.tensor -- The negative log-likelihood loss.
        """
        return nll_pc_hazard_loss(phi, idx_durations, events, interval_frac, self.reduction)


class DeepHitSingleLoss(_Loss):
    """Loss for deephit (single risk) model.
    Alpha is  weighting between likelihood and rank loss (so not like in paper):

    loss = alpha * nll + (1 - alpha) rank_loss(sigma)
    
    Arguments:
        alpha {float} -- Weighting between likelihood and rank loss.
        sigma {float} -- Part of rank loss (see DeepHit paper)

    Keyword Arguments:
        reduction {string} -- How to reduce the loss.
            'none': No reduction.
            'mean': Mean of tensor.
            'sum': sum.
    """
    def __init__(self, alpha, sigma, reduction='mean'):
        super().__init__(reduction)
        self.alpha = alpha
        self.sigma = sigma

    def forward(self, phi, idx_durations, events, rank_mat):
        nll = nll_pmf(phi, idx_durations, events, self.reduction)
        rank_loss = rank_loss_deephit_single(phi, idx_durations, events, rank_mat, self.sigma,
                                             self.reduction)
        return self.alpha * nll + (1. - self.alpha) * rank_loss


class DeepHitLoss(_Loss):
    """Loss for deephit model.
    If you have only one event type, use LossDeepHitSingle instead!

    Alpha is  weighting between likelihood and rank loss (so not like in paper):

    loss = alpha * nll + (1 - alpha) rank_loss(sigma)
    
    Arguments:
        alpha {float} -- Weighting between likelihood and rank loss.
        sigma {float} -- Part of rank loss (see DeepHit paper)
    """
    def __init__(self, alpha, sigma, reduction='mean'):
        super().__init__(reduction)
        self.alpha = alpha
        self.sigma = sigma

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        if (alpha < 0) or (alpha > 1):
            raise ValueError(f"Need `alpha` to be in [0, 1]. Got {alpha}.")
        self._alpha = alpha

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, sigma):
        if sigma <= 0:
            raise ValueError(f"Need `sigma` to be positive. Got {sigma}.")
        self._sigma = sigma

    def forward(self, phi, idx_durations, events, rank_mat):
        nll =  nll_pmf_cr(phi, idx_durations, events, self.reduction)
        rank_loss = rank_loss_deephit_cr(phi, idx_durations, events, rank_mat, self.sigma, self.reduction)
        return self.alpha * nll + (1. - self.alpha) * rank_loss


class CoxCCLoss(torch.nn.Module):
    """Torch loss functin for the Cox case-control models.

    loss_func = LossCoxCC()
    loss = loss_func(g_case, g_control)
    
    Keyword Arguments:
        shrink {float} -- Shinkage that encourage the net got give g_case and g_control
            closer to zero (a regularizer in a sense). (default: {0.})
        clamp {tuple} -- See code (default: {(-3e+38, 80.)})
    """
    def __init__(self, shrink=0., clamp=(-3e+38, 80.)):
        super().__init__()
        self.shrink = shrink
        self.clamp = clamp

    @property
    def shrink(self):
        return self._shrink
    
    @shrink.setter
    def shrink(self, shrink):
        if shrink < 0:
            raise ValueError(f"Need shrink to be non-negative, got {shrink}.")
        self._shrink = shrink

    def forward(self, g_case, g_control):
        return cox_cc_loss(g_case, g_control, self.shrink, self.clamp)


class CoxPHLoss(torch.nn.Module):
    """Loss for CoxPH.
    Requires the input to be sorted by descending duration time.
    See DatasetDurationSorted.

    We calculate the negative log of $(\frac{h_i}{\sum_{j \in R_i} h_j})^d$,
    where h = exp(log_h) are the hazards and R is the risk set, and d is event.

    We just compute a cumulative sum, and not the true Risk sets. This is a
    limitiation, but simple and fast.
    """
    def __init__(self):
        super().__init__()

    def forward(self, log_h, events):
        return cox_ph_loss(log_h, events)
