from typing import Tuple
import torch
from torch import Tensor
import torch.nn.functional as F
from pycox.models import utils
from torchtuples import TupleTree


def _reduction(loss: Tensor, reduction: str = 'mean') -> Tensor:
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    raise ValueError(f"`reduction` = {reduction} is not valid. Use 'none', 'mean' or 'sum'.")

def nll_logistic_hazard(phi: Tensor, idx_durations: Tensor, events: Tensor,
                        reduction: str = 'mean') -> Tensor:
    """Negative log-likelihood of the discrete time hazard parametrized model LogisticHazard [1].
    
    Arguments:
        phi {torch.tensor} -- Estimates in (-inf, inf), where hazard = sigmoid(phi).
        idx_durations {torch.tensor} -- Event times represented as indices.
        events {torch.tensor} -- Indicator of event (1.) or censoring (0.).
            Same length as 'idx_durations'.
        reduction {string} -- How to reduce the loss.
            'none': No reduction.
            'mean': Mean of tensor.
            'sum: sum.
    
    Returns:
        torch.tensor -- The negative log-likelihood.

    References:
    [1] Håvard Kvamme and Ørnulf Borgan. Continuous and Discrete-Time Survival Prediction
        with Neural Networks. arXiv preprint arXiv:1910.06724, 2019.
        https://arxiv.org/pdf/1910.06724.pdf
    """
    if phi.shape[1] <= idx_durations.max():
        raise ValueError(f"Network output `phi` is too small for `idx_durations`."+
                         f" Need at least `phi.shape[1] = {idx_durations.max().item()+1}`,"+
                         f" but got `phi.shape[1] = {phi.shape[1]}`")
    if events.dtype is torch.bool:
        events = events.float()
    events = events.view(-1, 1)
    idx_durations = idx_durations.view(-1, 1)
    y_bce = torch.zeros_like(phi).scatter(1, idx_durations, events)
    bce = F.binary_cross_entropy_with_logits(phi, y_bce, reduction='none')
    loss = bce.cumsum(1).gather(1, idx_durations).view(-1)
    return _reduction(loss, reduction)

def nll_pmf(phi: Tensor, idx_durations: Tensor, events: Tensor, reduction: str = 'mean',
            epsilon: float = 1e-7) -> Tensor:
    """Negative log-likelihood for the PMF parametrized model [1].
    
    Arguments:
        phi {torch.tensor} -- Estimates in (-inf, inf), where pmf = somefunc(phi).
        idx_durations {torch.tensor} -- Event times represented as indices.
        events {torch.tensor} -- Indicator of event (1.) or censoring (0.).
            Same length as 'idx_durations'.
        reduction {string} -- How to reduce the loss.
            'none': No reduction.
            'mean': Mean of tensor.
            'sum: sum.
    
    Returns:
        torch.tensor -- The negative log-likelihood.

    References:
    [1] Håvard Kvamme and Ørnulf Borgan. Continuous and Discrete-Time Survival Prediction
        with Neural Networks. arXiv preprint arXiv:1910.06724, 2019.
        https://arxiv.org/pdf/1910.06724.pdf
    """
    if phi.shape[1] <= idx_durations.max():
        raise ValueError(f"Network output `phi` is too small for `idx_durations`."+
                         f" Need at least `phi.shape[1] = {idx_durations.max().item()+1}`,"+
                         f" but got `phi.shape[1] = {phi.shape[1]}`")
    if events.dtype is torch.bool:
        events = events.float()
    events = events.view(-1)
    idx_durations = idx_durations.view(-1, 1)
    phi = utils.pad_col(phi)
    gamma = phi.max(1)[0]
    cumsum = phi.sub(gamma.view(-1, 1)).exp().cumsum(1)
    sum_ = cumsum[:, -1]
    part1 = phi.gather(1, idx_durations).view(-1).sub(gamma).mul(events)
    part2 = - sum_.relu().add(epsilon).log()
    part3 = sum_.sub(cumsum.gather(1, idx_durations).view(-1)).relu().add(epsilon).log().mul(1. - events)
    # need relu() in part3 (and possibly part2) because cumsum on gpu has some bugs and we risk getting negative numbers.
    loss = - part1.add(part2).add(part3)
    return _reduction(loss, reduction)

def nll_mtlr(phi: Tensor, idx_durations: Tensor, events: Tensor, reduction: str = 'mean',
             epsilon: float = 1e-7) -> Tensor:
    """Negative log-likelihood for the MTLR parametrized model [1] [2].

    This is essentially a PMF parametrization with an extra cumulative sum, as explained in [3].
    
    Arguments:
        phi {torch.tensor} -- Estimates in (-inf, inf), where pmf = somefunc(phi).
        idx_durations {torch.tensor} -- Event times represented as indices.
        events {torch.tensor} -- Indicator of event (1.) or censoring (0.).
            Same length as 'idx_durations'.
        reduction {string} -- How to reduce the loss.
            'none': No reduction.
            'mean': Mean of tensor.
            'sum: sum.
    
    Returns:
        torch.tensor -- The negative log-likelihood.

    References:
    [1] Chun-Nam Yu, Russell Greiner, Hsiu-Chin Lin, and Vickie Baracos.
        Learning patient- specific cancer survival distributions as a sequence of dependent regressors.
        In Advances in Neural Information Processing Systems 24, pages 1845–1853.
        Curran Associates, Inc., 2011.
        https://papers.nips.cc/paper/4210-learning-patient-specific-cancer-survival-distributions-as-a-sequence-of-dependent-regressors.pdf

    [2] Stephane Fotso. Deep neural networks for survival analysis based on a multi-task framework.
        arXiv preprint arXiv:1801.05512, 2018.
        https://arxiv.org/pdf/1801.05512.pdf

    [3] Håvard Kvamme and Ørnulf Borgan. Continuous and Discrete-Time Survival Prediction
        with Neural Networks. arXiv preprint arXiv:1910.06724, 2019.
        https://arxiv.org/pdf/1910.06724.pdf
    """
    phi = utils.cumsum_reverse(phi, dim=1)
    return nll_pmf(phi, idx_durations, events, reduction, epsilon)

def nll_pc_hazard_loss(phi: Tensor, idx_durations: Tensor, events: Tensor, interval_frac: Tensor,
                       reduction: str = 'mean') -> Tensor:
    """Negative log-likelihood of the PC-Hazard parametrization model [1].
    
    Arguments:
        phi {torch.tensor} -- Estimates in (-inf, inf), where hazard = sigmoid(phi).
        idx_durations {torch.tensor} -- Event times represented as indices.
        events {torch.tensor} -- Indicator of event (1.) or censoring (0.).
            Same length as 'idx_durations'.
        interval_frac {torch.tensor} -- Fraction of last interval before event/censoring.
        reduction {string} -- How to reduce the loss.
            'none': No reduction.
            'mean': Mean of tensor.
            'sum: sum.
    
    Returns:
        torch.tensor -- The negative log-likelihood.

    References:
    [1] Håvard Kvamme and Ørnulf Borgan. Continuous and Discrete-Time Survival Prediction
        with Neural Networks. arXiv preprint arXiv:1910.06724, 2019.
        https://arxiv.org/pdf/1910.06724.pdf
    """
    if events.dtype is torch.bool:
        events = events.float()
    idx_durations = idx_durations.view(-1, 1)
    events = events.view(-1)
    interval_frac = interval_frac.view(-1)

    keep = idx_durations.view(-1) >= 0
    phi = phi[keep, :]
    idx_durations = idx_durations[keep, :]
    events = events[keep]
    interval_frac = interval_frac[keep]

    # log_h_e = F.softplus(phi.gather(1, idx_durations).view(-1)).log().mul(events)
    log_h_e = utils.log_softplus(phi.gather(1, idx_durations).view(-1)).mul(events)
    haz = F.softplus(phi)
    scaled_h_e = haz.gather(1, idx_durations).view(-1).mul(interval_frac)
    haz = utils.pad_col(haz, where='start')
    sum_haz = haz.cumsum(1).gather(1, idx_durations).view(-1) 
    loss = - log_h_e.sub(scaled_h_e).sub(sum_haz)
    return _reduction(loss, reduction)


def _rank_loss_deephit(pmf: Tensor, y: Tensor, rank_mat: Tensor, sigma: float,
                       reduction: str = 'mean') -> Tensor:
    """Ranking loss from DeepHit.
    
    Arguments:
        pmf {torch.tensor} -- Matrix with probability mass function pmf_ij = f_i(t_j)
        y {torch.tensor} -- Matrix with indicator of duration and censoring time. 
        rank_mat {torch.tensor} -- See pair_rank_mat function.
        sigma {float} -- Sigma from DeepHit paper, chosen by you.
    
    Returns:
        torch.tensor -- loss
    """
    r = _diff_cdf_at_time_i(pmf, y)
    loss = rank_mat * torch.exp(-r/sigma)
    loss = loss.mean(1, keepdim=True)
    return _reduction(loss, reduction)

def _diff_cdf_at_time_i(pmf: Tensor, y: Tensor) -> Tensor:
    """R is the matrix from the DeepHit code giving the difference in CDF between individual
    i and j, at the event time of j. 
    I.e: R_ij = F_i(T_i) - F_j(T_i)
    
    Arguments:
        pmf {torch.tensor} -- Matrix with probability mass function pmf_ij = f_i(t_j)
        y {torch.tensor} -- Matrix with indicator of duration/censor time.
    
    Returns:
        torch.tensor -- R_ij = F_i(T_i) - F_j(T_i)
    """
    n = pmf.shape[0]
    ones = torch.ones((n, 1), device=pmf.device)
    r = pmf.cumsum(1).matmul(y.transpose(0, 1))
    diag_r = r.diag().view(1, -1)
    r = ones.matmul(diag_r) - r
    return r.transpose(0, 1)

def rank_loss_deephit_single(phi: Tensor, idx_durations: Tensor, events: Tensor, rank_mat: Tensor,
                             sigma: Tensor, reduction: str = 'mean') -> Tensor:
    """Rank loss proposed by DeepHit authors [1] for a single risks.
    
    Arguments:
        pmf {torch.tensor} -- Matrix with probability mass function pmf_ij = f_i(t_j)
        y {torch.tensor} -- Matrix with indicator of duration and censoring time. 
        rank_mat {torch.tensor} -- See pair_rank_mat function.
        sigma {float} -- Sigma from DeepHit paper, chosen by you.
    Arguments:
        phi {torch.tensor} -- Predictions as float tensor with shape [batch, n_durations]
            all in (-inf, inf).
        idx_durations {torch.tensor} -- Int tensor with index of durations.
        events {torch.tensor} -- Float indicator of event or censoring (1 is event).
        rank_mat {torch.tensor} -- See pair_rank_mat function.
        sigma {float} -- Sigma from DeepHit paper, chosen by you.
    
    Keyword Arguments:
        reduction {string} -- How to reduce the loss.
            'none': No reduction.
            'mean': Mean of tensor.
            'sum': sum.
    
    Returns:
        torch.tensor -- Rank loss.

    References:
    [1] Changhee Lee, William R Zame, Jinsung Yoon, and Mihaela van der Schaar. Deephit: A deep learning
        approach to survival analysis with competing risks. In Thirty-Second AAAI Conference on Artificial
        Intelligence, 2018.
        http://medianetlab.ee.ucla.edu/papers/AAAI_2018_DeepHit
    """
    idx_durations = idx_durations.view(-1, 1)
    # events = events.float().view(-1)
    pmf = utils.pad_col(phi).softmax(1)
    y = torch.zeros_like(pmf).scatter(1, idx_durations, 1.) # one-hot
    rank_loss = _rank_loss_deephit(pmf, y, rank_mat, sigma, reduction)
    return rank_loss

def nll_pmf_cr(phi: Tensor, idx_durations: Tensor, events: Tensor, reduction: str = 'mean',
               epsilon: float = 1e-7) -> Tensor:
    """Negative log-likelihood for PMF parameterizations. `phi` is the ''logit''.
    
    Arguments:
        phi {torch.tensor} -- Predictions as float tensor with shape [batch, n_risks, n_durations]
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
    # Should improve numerical stability by, e.g., log-sum-exp trick.
    events = events.view(-1) - 1
    event_01 = (events != -1).float()
    idx_durations = idx_durations.view(-1)
    batch_size = phi.size(0)
    sm = utils.pad_col(phi.view(batch_size, -1)).softmax(1)[:, :-1].view(phi.shape)
    index = torch.arange(batch_size)
    part1 = sm[index, events, idx_durations].relu().add(epsilon).log().mul(event_01)
    part2 = (1 - sm.cumsum(2)[index, :, idx_durations].sum(1)).relu().add(epsilon).log().mul(1 - event_01)
    loss = - part1.add(part2)
    return _reduction(loss, reduction)

def rank_loss_deephit_cr(phi: Tensor, idx_durations: Tensor, events: Tensor, rank_mat: Tensor,
                         sigma: float, reduction: str = 'mean') -> Tensor:
    """Rank loss proposed by DeepHit authors for competing risks [1].
    
    Arguments:
        phi {torch.tensor} -- Predictions as float tensor with shape [batch, n_risks, n_durations]
            all in (-inf, inf).
        idx_durations {torch.tensor} -- Int tensor with index of durations.
        events {torch.tensor} -- Int tensor with event types.
            {0: Censored, 1: first group, ..., n_risks: n'th risk group}.
        rank_mat {torch.tensor} -- See pair_rank_mat function.
        sigma {float} -- Sigma from DeepHit paper, chosen by you.
    
    Keyword Arguments:
        reduction {string} -- How to reduce the loss.
            'none': No reduction.
            'mean': Mean of tensor.
            else: sum.
    
    Returns:
        torch.tensor -- Rank loss.

    References:
    [1] Changhee Lee, William R Zame, Jinsung Yoon, and Mihaela van der Schaar. Deephit: A deep learning
        approach to survival analysis with competing risks. In Thirty-Second AAAI Conference on Artificial
        Intelligence, 2018.
        http://medianetlab.ee.ucla.edu/papers/AAAI_2018_DeepHit
    """
    idx_durations = idx_durations.view(-1)
    events = events.view(-1) - 1
    event_01 = (events == -1).float()

    batch_size, n_risks = phi.shape[:2]
    pmf = utils.pad_col(phi.view(batch_size, -1)).softmax(1)
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

def bce_surv_loss(phi: Tensor, idx_durations: Tensor, events: Tensor, reduction: str = 'mean') -> Tensor:
    """Loss function for a set of binary classifiers. Each output node (element in `phi`)
    is the logit of a survival prediction at the time corresponding to that index.
    See [1] for explanation of the method.
    
    Arguments:
        phi {torch.tensor} -- Estimates in (-inf, inf), where survival = sigmoid(phi).
        idx_durations {torch.tensor} -- Event times represented as indices.
        events {torch.tensor} -- Indicator of event (1.) or censoring (0.).
            Same length as 'idx_durations'.
    
    Keyword Arguments:
        reduction {string} -- How to reduce the loss.
            'none': No reduction.
            'mean': Mean of tensor.
            'sum: sum.

    References:
        [1] Håvard Kvamme and Ørnulf Borgan. The Brier Score under Administrative Censoring: Problems
            and Solutions. arXiv preprint arXiv:1912.08581, 2019.
            https://arxiv.org/pdf/1912.08581.pdf
    """
    if phi.shape[1] <= idx_durations.max():
        raise ValueError(f"Network output `phi` is too small for `idx_durations`."+
                         f" Need at least `phi.shape[1] = {idx_durations.max().item()+1}`,"+
                         f" but got `phi.shape[1] = {phi.shape[1]}`")
    if events.dtype is torch.bool:
        events = events.float()
    y = torch.arange(phi.shape[1], dtype=idx_durations.dtype, device=idx_durations.device)
    y = (y.view(1, -1) < idx_durations.view(-1, 1)).float() # mask with ones until idx_duration
    c = y + (torch.ones_like(y) - y) * events.view(-1, 1)  # mask with ones until censoring.
    return F.binary_cross_entropy_with_logits(phi, y, c, reduction=reduction)

def cox_cc_loss(g_case: Tensor, g_control: Tensor, shrink : float = 0.,
                clamp: Tuple[float, float] = (-3e+38, 80.)) -> Tensor:
    """Torch loss function for the Cox case-control models.
    For only one control, see `cox_cc_loss_single_ctrl` instead.
    
    Arguments:
        g_case {torch.Tensor} -- Result of net(input_case)
        g_control {torch.Tensor} -- Results of [net(input_ctrl1), net(input_ctrl2), ...]
    
    Keyword Arguments:
        shrink {float} -- Shrinkage that encourage the net got give g_case and g_control
            closer to zero (a regularizer in a sense). (default: {0.})
        clamp {tuple} -- See code (default: {(-3e+38, 80.)})
    
    Returns:
        [type] -- [description]
    """
    control_sum = 0.
    shrink_control = 0.
    if g_case.shape != g_control[0].shape:
        raise ValueError(f"Need `g_case` and `g_control[0]` to have same shape. Got {g_case.shape}"+
                         f" and {g_control[0].shape}")
    for ctr in g_control:
        shrink_control += ctr.abs().mean()
        ctr = ctr - g_case
        ctr = torch.clamp(ctr, *clamp)  # Kills grads for very bad cases (should instead cap grads!!!).
        control_sum += torch.exp(ctr)
    loss = torch.log(1. + control_sum)
    shrink_zero = shrink * (g_case.abs().mean() + shrink_control) / len(g_control)
    return torch.mean(loss) + shrink_zero.abs()

def cox_cc_loss_single_ctrl(g_case: Tensor, g_control: Tensor, shrink: float = 0.) -> Tensor:
    """CoxCC and CoxTime loss, but with only a single control.
    """
    loss = F.softplus(g_control - g_case).mean()
    if shrink != 0:
        loss += shrink * (g_case.abs().mean() + g_control.abs().mean())
    return loss

def cox_ph_loss_sorted(log_h: Tensor, events: Tensor, eps: float = 1e-7) -> Tensor:
    """Requires the input to be sorted by descending duration time.
    See DatasetDurationSorted.

    We calculate the negative log of $(\frac{h_i}{\sum_{j \in R_i} h_j})^d$,
    where h = exp(log_h) are the hazards and R is the risk set, and d is event.

    We just compute a cumulative sum, and not the true Risk sets. This is a
    limitation, but simple and fast.
    """
    if events.dtype is torch.bool:
        events = events.float()
    events = events.view(-1)
    log_h = log_h.view(-1)
    gamma = log_h.max()
    log_cumsum_h = log_h.sub(gamma).exp().cumsum(0).add(eps).log().add(gamma)
    return - log_h.sub(log_cumsum_h).mul(events).sum().div(events.sum())

def cox_ph_loss(log_h: Tensor, durations: Tensor, events: Tensor, eps: float = 1e-7) -> Tensor:
    """Loss for CoxPH model. If data is sorted by descending duration, see `cox_ph_loss_sorted`.

    We calculate the negative log of $(\frac{h_i}{\sum_{j \in R_i} h_j})^d$,
    where h = exp(log_h) are the hazards and R is the risk set, and d is event.

    We just compute a cumulative sum, and not the true Risk sets. This is a
    limitation, but simple and fast.
    """
    idx = durations.sort(descending=True)[1]
    events = events[idx]
    log_h = log_h[idx]
    return cox_ph_loss_sorted(log_h, events, eps)


class _Loss(torch.nn.Module):
    """Generic loss function.
    
    Arguments:
        reduction {string} -- How to reduce the loss.
            'none': No reduction.
            'mean': Mean of tensor.
            'sum: sum.
    """
    def __init__(self, reduction: str = 'mean') -> None:
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
    def forward(self, phi: Tensor, idx_durations: Tensor, events: Tensor) -> Tensor:
        return nll_logistic_hazard(phi, idx_durations, events, self.reduction)


class NLLPMFLoss(_Loss):
    """Negative log-likelihood of the PMF parametrization model.
    See `loss.nll_pmf` for details.
    
    Arguments:
        reduction {string} -- How to reduce the loss.
            'none': No reduction.
            'mean': Mean of tensor.
            'sum: sum.
    
    Returns:
        torch.tensor -- The negative log-likelihood.
    """
    def forward(self, phi: Tensor, idx_durations: Tensor, events: Tensor) -> Tensor:
        return nll_pmf(phi, idx_durations, events, self.reduction)


class NLLMTLRLoss(_Loss):
    """Negative log-likelihood for the MTLR parametrized model.
    See `loss.nll_mtlr` for details.

    This is essentially a PMF parametrization with an extra cumulative sum.
    See [paper link] for an explanation.
    
    Arguments:
        reduction {string} -- How to reduce the loss.
            'none': No reduction.
            'mean': Mean of tensor.
            'sum: sum.
    
    Returns:
        torch.tensor -- The negative log-likelihood.
    """
    def forward(self, phi: Tensor, idx_durations: Tensor, events: Tensor) -> Tensor:
        return nll_mtlr(phi, idx_durations, events, self.reduction)


class NLLPCHazardLoss(_Loss):
    def forward(self, phi: Tensor, idx_durations: Tensor, events: Tensor, interval_frac: Tensor,
                reduction: str = 'mean') -> Tensor:
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


class _DeepHitLoss(_Loss):
    """Loss for DeepHit model.
    If you have only one event type, use LossDeepHitSingle instead!

    Alpha is  weighting between likelihood and rank loss (so not like in paper):

    loss = alpha * nll + (1 - alpha) rank_loss(sigma)
    
    Arguments:
        alpha {float} -- Weighting between likelihood and rank loss.
        sigma {float} -- Part of rank loss (see DeepHit paper)
    """
    def __init__(self, alpha: float, sigma: float, reduction: str = 'mean') -> None:
        super().__init__(reduction)
        self.alpha = alpha
        self.sigma = sigma

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, alpha: float) -> None:
        if (alpha < 0) or (alpha > 1):
            raise ValueError(f"Need `alpha` to be in [0, 1]. Got {alpha}.")
        self._alpha = alpha

    @property
    def sigma(self) -> float:
        return self._sigma

    @sigma.setter
    def sigma(self, sigma: float) -> None:
        if sigma <= 0:
            raise ValueError(f"Need `sigma` to be positive. Got {sigma}.")
        self._sigma = sigma


class DeepHitSingleLoss(_DeepHitLoss):
    """Loss for DeepHit (single risk) model [1].
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

    References:
    [1] Changhee Lee, William R Zame, Jinsung Yoon, and Mihaela van der Schaar. Deephit: A deep learning
        approach to survival analysis with competing risks. In Thirty-Second AAAI Conference on Artificial
        Intelligence, 2018.
        http://medianetlab.ee.ucla.edu/papers/AAAI_2018_DeepHit
    """
    def forward(self, phi: Tensor, idx_durations: Tensor, events: Tensor, rank_mat: Tensor) -> Tensor:
        nll = nll_pmf(phi, idx_durations, events, self.reduction)
        rank_loss = rank_loss_deephit_single(phi, idx_durations, events, rank_mat, self.sigma,
                                             self.reduction)
        return self.alpha * nll + (1. - self.alpha) * rank_loss


class DeepHitLoss(_DeepHitLoss):
    """Loss for DeepHit model [1].
    If you have only one event type, use LossDeepHitSingle instead!

    Alpha is  weighting between likelihood and rank loss (so not like in paper):

    loss = alpha * nll + (1 - alpha) rank_loss(sigma)

    Arguments:
        alpha {float} -- Weighting between likelihood and rank loss.
        sigma {float} -- Part of rank loss (see DeepHit paper)

    References:
    [1] Changhee Lee, William R Zame, Jinsung Yoon, and Mihaela van der Schaar. Deephit: A deep learning
        approach to survival analysis with competing risks. In Thirty-Second AAAI Conference on Artificial
        Intelligence, 2018.
        http://medianetlab.ee.ucla.edu/papers/AAAI_2018_DeepHit
    """
    def forward(self, phi: Tensor, idx_durations: Tensor, events: Tensor, rank_mat: Tensor) -> Tensor:
        nll =  nll_pmf_cr(phi, idx_durations, events, self.reduction)
        rank_loss = rank_loss_deephit_cr(phi, idx_durations, events, rank_mat, self.sigma, self.reduction)
        return self.alpha * nll + (1. - self.alpha) * rank_loss


class BCESurvLoss(_Loss):
    """Loss function of the BCESurv method.
    See `loss.bce_surv_loss` for details.

    Arguments:
        reduction {string} -- How to reduce the loss.
            'none': No reduction.
            'mean': Mean of tensor.
            'sum: sum.

    Returns:
        torch.tensor -- The negative log-likelihood.
    """
    def forward(self, phi: Tensor, idx_durations: Tensor, events: Tensor) -> Tensor:
        return bce_surv_loss(phi, idx_durations, events, self.reduction)


class CoxCCLoss(torch.nn.Module):
    """Torch loss function for the Cox case-control models.

    loss_func = LossCoxCC()
    loss = loss_func(g_case, g_control)
    
    Keyword Arguments:
        shrink {float} -- Shrinkage that encourage the net got give g_case and g_control
            closer to zero (a regularizer in a sense). (default: {0.})
        clamp {tuple} -- See code (default: {(-3e+38, 80.)})
    """
    def __init__(self, shrink: float = 0., clamp: Tuple[float, float] = (-3e+38, 80.)) -> Tensor:
        super().__init__()
        self.shrink = shrink
        self.clamp = clamp

    @property
    def shrink(self) -> float:
        return self._shrink
    
    @shrink.setter
    def shrink(self, shrink: float) -> None:
        if shrink < 0:
            raise ValueError(f"Need shrink to be non-negative, got {shrink}.")
        self._shrink = shrink

    def forward(self, g_case: Tensor, g_control: TupleTree) -> Tensor:
        single = False
        if hasattr(g_control, 'shape'):
             if g_case.shape == g_control.shape:
                return cox_cc_loss_single_ctrl(g_case, g_control, self.shrink)
        elif (len(g_control) == 1) and (g_control[0].shape == g_case.shape):
                return cox_cc_loss_single_ctrl(g_case, g_control[0], self.shrink)
        return cox_cc_loss(g_case, g_control, self.shrink, self.clamp)


class CoxPHLossSorted(torch.nn.Module):
    """Loss for CoxPH.
    Requires the input to be sorted by descending duration time.
    See DatasetDurationSorted.

    We calculate the negative log of $(\frac{h_i}{\sum_{j \in R_i} h_j})^d$,
    where h = exp(log_h) are the hazards and R is the risk set, and d is event.

    We just compute a cumulative sum, and not the true Risk sets. This is a
    limitation, but simple and fast.
    """
    def __init__(self):
        super().__init__()

    def forward(self, log_h: Tensor, events: Tensor) -> Tensor:
        return cox_ph_loss_sorted(log_h, events)


class CoxPHLoss(torch.nn.Module):
    """Loss for CoxPH model. If data is sorted by descending duration, see `cox_ph_loss_sorted`.

    We calculate the negative log of $(\frac{h_i}{\sum_{j \in R_i} h_j})^d$,
    where h = exp(log_h) are the hazards and R is the risk set, and d is event.

    We just compute a cumulative sum, and not the true Risk sets. This is a
    limitation, but simple and fast.
    """
    def forward(self, log_h: Tensor, durations: Tensor, events: Tensor) -> Tensor:
        return cox_ph_loss(log_h, durations, events)

