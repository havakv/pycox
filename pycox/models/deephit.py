"""Impleentation of the deephit model, but currently for single event cause
(not compteeting risk).
"""
import numpy as np
import numba
import torch
from torchtuples import Model, tuplefy
from torchtuples.data import DatasetTuple


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
            else: sum.
    
    Returns:
        torch.tensor -- Retruns mean negative log-likelihood.
    """
    if (idx_durations.max()) >= phi.shape[1]:
        raise ValueError("""'t_idx' too large. Probably need to increase output size of net.""")
    events = events.view(-1)
    idx_durations = idx_durations.view(-1, 1)
    phi = _pad_phi(phi)
    gamma = phi.max(1)[0]
    cumsum = phi.sub(gamma.view(-1, 1)).exp().cumsum(1)
    sum_ = cumsum[:, -1]
    part1 = phi.gather(1, idx_durations).view(-1).sub(gamma).mul(events)
    part2 = - sum_.relu().add(epsilon).log()
    part3 = sum_.sub(cumsum.gather(1, idx_durations).view(-1)).relu().add(epsilon).log().mul(1. - events)
    # need relu() in part3 (and possibly part2) because cumsum on gpu has some bugs and we risk getting negative numbers.
    loss = - part1.add(part2).add(part3)
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    return loss.sum()

def _pad_phi(phi):
    """Addes a column of zeros at the end of phi"""
    phi_mp1 = torch.zeros_like(phi[:, :1])
    return torch.cat([phi, phi_mp1], dim=1)

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
    eta = rank_mat * torch.exp(-r/sigma)
    eta = eta.mean(1, keepdim=True)
    if reduction == 'none':
        return eta
    elif reduction == 'mean':
        return eta.mean()
    return eta.sum()

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
            else: sum.
    
    Returns:
        torch.tensor -- Rank loss.
    """
    idx_durations = idx_durations.view(-1, 1)
    events = events.float().view(-1)
    pmf = _pad_phi(phi).softmax(1)
    y = torch.zeros_like(pmf).scatter(1, idx_durations, 1.) # one-hot
    rank_loss = _rank_loss_deephit(pmf, y, rank_mat, sigma, reduction)
    return rank_loss


class LossDeepHitSingle(torch.nn.Module):
    """Loss for deephit (single risk) model.
    Alpha is  weighting between likelihood and rank loss (so not like in paper):

    loss = alpha * nll + (1 - alpha) rank_loss(sigma)
    
    Arguments:
        alpha {float} -- Weighting between likelihood and rank loss.
        sigma {float} -- Part of rank loss (see DeepHit paper)
    """
    def __init__(self, alpha, sigma):
        super().__init__()
        self.alpha = alpha
        self.sigma = sigma

    def forward(self, phi, idx_durations, events, rank_mat):
        nll = nll_pmf(phi, idx_durations, events)
        rank_loss = rank_loss_deephit_single(phi, idx_durations, events, rank_mat, self.sigma)
        # pmf = _pad_phi(phi).softmax(1)
        # y = torch.zeros_like(pmf).scatter(1, idx_durations.view(-1, 1), 1.) # one-hot
        # rank_loss = _rank_loss_deephit(pmf, y, rank_mat, self.sigma)
        return self.alpha * nll + (1. - self.alpha) * rank_loss


class DeepHitSingle(Model):
    """Essentailly same as torchtuples.Model, but instead of specifying a loss function,
    we now specify 'alpha' and 'simga'.
    See paper (or make_loss_deephit function)
    
    Keyword Arguments:
        alpha {float} -- Weighting (0, 1) likelihood and rank loss (L2 in paper).
            1 gives only likelihood, and 0 gives only rank loss. (default: {0.5})
        sigma {float} -- from eta in rank loss (L2 in paper) (default: {0.1})
    """
    def __init__(self, net, optimizer=None, device=None, alpha=0.5, sigma=0.1):
        loss = LossDeepHitSingle(alpha, sigma)
        super().__init__(net, loss, optimizer, device)

    def make_dataloader(self, data, batch_size, shuffle, num_workers=0):
        dataloader = super().make_dataloader(data, batch_size, shuffle, num_workers,
                                             make_dataset=DatasetDeepHitSingle)
        return dataloader
    
    def make_dataloader_predict(self, input, batch_size, shuffle=False, num_workers=0):
        dataloader = super().make_dataloader(input, batch_size, shuffle, num_workers)
        return dataloader

    def predict_survival_function(self, input, batch_size=8224, eval_=True, to_cpu=False,
                                  num_workers=0):
        """Might need to set to_cpu to true if too large dataset."""
        pmf = self.predict_pmf(input, batch_size, eval_, to_cpu, num_workers, False)
        surv = 1 - pmf.cumsum(0)
        if tuplefy(input).type() is np.ndarray:
            surv = surv.cpu().numpy()
        return surv

    def predict_pmf(self, input, batch_size=8224, eval_=True, to_cpu=False, num_workers=0,
                    numpy=None):
        """Mighht need to set to_cpu to true if too large dataset."""
        preds = self.predict(input, batch_size, False, eval_, False, to_cpu, num_workers)
        pmf = _pad_phi(preds).softmax(1)[:, :-1].transpose(0, 1)
        if numpy is None:
            if tuplefy(input).type() is np.ndarray:
                pmf = pmf.cpu().numpy()
        elif numpy is True:
                pmf = pmf.cpu().numpy()
        return pmf


class DatasetDeepHitSingle(DatasetTuple):
    def __getitem__(self, index):
        input, target =  super().__getitem__(index)
        target = target.to_numpy()
        rank_mat = pair_rank_mat(*target)
        target = tuplefy(*target, rank_mat).to_tensor()
        return tuplefy(input, target)


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

@numba.jit(nopython=True)
def _pair_rank_mat(mat, idx_durations, events, dtype='float32'):
    n = len(idx_durations)
    for i in range(n):
        dur_i = idx_durations[i]
        ev_i = events[i]
        if ev_i == 0:
            continue
        for j in range(n):
            dur_j = idx_durations[j]
            ev_j = events[j]
            if (dur_i < dur_j) or ((dur_i == dur_j) and (ev_j == 0)):
                mat[i, j] = 1
    return mat

def pair_rank_mat(idx_durations, events, dtype='float32'):
    """Indicator matrix R with R_ij = 1{T_i < T_j and D_i = 1}.
    So it takes value 1 if we observe that i has an event before j and zero otherwise.
    
    Arguments:
        idx_durations {np.array} -- Array with durations.
        events {np.array} -- Array with event indicators.
    
    Keyword Arguments:
        dtype {str} -- dtype of array (default: {'float32'})
    
    Returns:
        np.array -- n x n matrix indicating if i has an observerd event before j.
    """
    idx_durations = idx_durations.reshape(-1)
    events = events.reshape(-1)
    n = len(idx_durations)
    mat = np.zeros((n, n), dtype=dtype)
    mat = _pair_rank_mat(mat, idx_durations, events, dtype)
    return mat

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
    sm = _pad_phi(phi.view(batch_size, -1)).softmax(1)[:, :-1].view(phi.shape)
    index = torch.arange(batch_size)
    part1 = sm[index, events, idx_durations].relu().add(epsilon).log().mul(event_01)
    part2 = (1 - sm.cumsum(2)[index, :, idx_durations].sum(1)).relu().add(epsilon).log().mul(1 - event_01)
    loss = - part1.add(part2)
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    return loss.sum()

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
    pmf = _pad_phi(phi.view(batch_size, -1)).softmax(1)
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
    return sum([lo.sum() for lo in loss])

class LossDeepHit(torch.nn.Module):
    """Loss for deephit model.
    If you have only one event type, use LossDeepHitSingle instead!

    Alpha is  weighting between likelihood and rank loss (so not like in paper):

    loss = alpha * nll + (1 - alpha) rank_loss(sigma)
    
    Arguments:
        alpha {float} -- Weighting between likelihood and rank loss.
        sigma {float} -- Part of rank loss (see DeepHit paper)
    """
    def __init__(self, alpha, sigma):
        super().__init__()
        self.alpha = alpha
        self.sigma = sigma

    def forward(self, phi, idx_durations, events, rank_mat):
        nll =  nll_pmf_cr(phi, idx_durations, events)
        rank_loss = rank_loss_deephit_cr(phi, idx_durations, events, rank_mat, self.sigma)
        return self.alpha * nll + (1. - self.alpha) * rank_loss


class DatasetDeepHit(DatasetDeepHitSingle):
    pass


class DeepHit(Model):
    """DeepHit for competing risks.
    If you have only one event type, use DeepHitSingle instead!

    Essentailly same as torchtuples.Model, but instead of specifying a loss function,
    we now specify 'alpha' and 'simga'.
    See paper (or make_loss_deephit function)
    
    Keyword Arguments:
        alpha {float} -- Weighting (0, 1) likelihood and rank loss (L2 in paper).
            1 gives only likelihood, and 0 gives only rank loss. (default: {0.5})
        sigma {float} -- from eta in rank loss (L2 in paper) (default: {0.1})
    """
    def __init__(self, net, optimizer=None, device=None, alpha=0.5, sigma=0.1):
        loss = LossDeepHit(alpha, sigma)
        super().__init__(net, loss, optimizer, device)

    def make_dataloader(self, data, batch_size, shuffle, num_workers=0):
        dataloader = super().make_dataloader(data, batch_size, shuffle, num_workers,
                                             make_dataset=DatasetDeepHit)
        return dataloader
    
    def make_dataloader_predict(self, input, batch_size, shuffle=False, num_workers=0):
        dataloader = super().make_dataloader(input, batch_size, shuffle, num_workers)
        return dataloader

    def predict_cif(self, input, batch_size=8224, eval_=True, to_cpu=False, num_workers=0,
                    numpy=None):
        pmf = self.predict_pmf(input, batch_size, eval_, to_cpu, num_workers, False)
        cif = pmf.cumsum(1)
        if numpy is None:
            if tuplefy(input).type() is np.ndarray:
                cif = cif.cpu().numpy()
        elif numpy is True:
                cif = cif.cpu().numpy()
        return cif

    def predict_survival_function(self, input, batch_size=8224, eval_=True, to_cpu=False,
                                  num_workers=0, numpy=None):
        """Might need to set to_cpu to true if too large dataset."""
        cif = self.predict_cif(input, batch_size, eval_, to_cpu, num_workers, False)
        surv = 1. - cif.sum(0)
        if numpy is None:
            if tuplefy(input).type() is np.ndarray:
                surv = surv.cpu().numpy()
        elif numpy is True:
                surv = surv.cpu().numpy()
        return surv

    def predict_pmf(self, input, batch_size=8224, eval_=True, to_cpu=False, num_workers=0,
                    numpy=None):
        """Mighht need to set to_cpu to true if too large dataset."""
        preds = self.predict(input, batch_size, False, eval_, False, to_cpu, num_workers)
        pmf = _pad_phi(preds.view(preds.size(0), -1)).softmax(1)[:, :-1]
        pmf = pmf.view(preds.shape).transpose(0, 1).transpose(1, 2)
        if numpy is None:
            if tuplefy(input).type() is np.ndarray:
                pmf = pmf.cpu().numpy()
        elif numpy is True:
                pmf = pmf.cpu().numpy()
        return pmf
