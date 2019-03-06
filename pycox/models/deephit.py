"""Impleentation of the deephit model, but currently for single event cause
(not compteeting risk).
"""
import numpy as np
import numba
import torch
from pyth import Model, tuplefy
from pyth.data import DatasetTuple


class DeepHitSingle(Model):
    """Essentailly same as pyth.Model, but instead of specifying a loss function,
    we now specify 'alpha' and 'simga'.
    See paper (or make_loss_deephit function)
    
    Keyword Arguments:
        alpha {float} -- Weighting (0, 1) likelihood and rank loss (L2 in paper).
            1 gives only likelihood, and 0 gives only rank loss. (default: {0.5})
        sigma {float} -- from eta in rank loss (L2 in paper) (default: {0.1})
    """
    def __init__(self, net, optimizer=None, device=None, alpha=0.5, sigma=0.1):
        loss = make_loss_deephit(alpha, sigma)
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
        """Mighht need to set to_cpu to true if too large dataset."""
        cdf = (self.predict(input, batch_size, False, eval_, to_cpu, num_workers)
               .softmax(1)
               [:, :-1]
               .cumsum(1)
               .cpu()
               .numpy())
        return 1 - cdf.transpose()


def make_loss_deephit(alpha, sigma):
    """Loss for deephit (single risk) model.
    Alpha is is weighting between likelihood and rank loss (so not like in paper):

    loss = alpha * nll + (1 - alpha) rank_loss(sigma)
    """
    rank_loss = make_rank_loss_deephit(sigma)
    # def loss(phi, y, d, not_e_c, rank_mat):
    def loss(phi, y, d, rank_mat):
        nll = nll_pmf(phi, y, d)
        pmf = phi.softmax(1)
        # r_loss = rank_loss(pmf, not_e_c, rank_mat)
        r_loss = rank_loss(pmf, y, rank_mat)
        return alpha * nll + (1. - alpha) * r_loss
    return loss

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

def make_rank_loss_deephit(sigma):
    """Make ranking loss funciton from deephit.
    
    Arguments:
        sigma {float} -- Sigma from paper.
    
    Returns:
        function -- loss function.
    """
    # def rank_loss_deephit(pmf, not_ec, rank_mat, reduction='elementwise_mean'):
    def rank_loss_deephit(pmf, y, rank_mat, reduction='elementwise_mean'):
        """Ranking loss from deephit paper.
        
        Arguments:
            pmf {torch.tensor} -- Matrix with probability mass function pmf_ij = f_i(t_j)
            y {torch.tensor} -- Matrix with indicator of duration/censoring time. 

            # not_ec {torch.tensor} -- Indicator matrix with same shape as 'pmf', indicating
            #     that the individual has still not had an event or censoring.
            # rank_mat {torch.tensor} -- See pair_rank_mat function.
        
        Returns:
            torch.tensor -- loss
        """
        # r = diff_cdf_at_time_i(pmf, not_ec)
        r = diff_cdf_at_time_i(pmf, y)
        eta = rank_mat * torch.exp(-r/sigma)
        eta = eta.mean(1, keepdim=True)
        if reduction == 'none':
            return eta
        elif reduction == 'elementwise_mean':
            return eta.mean()
        return eta.sum()
        # return eta
    return rank_loss_deephit

# def diff_cdf_at_time_i(pmf, not_ec):
def diff_cdf_at_time_i(pmf, y):
    """R is the matrix from the deephit code giving the difference in cdf between individual
    i and j, at the event time of j. 
    I.e: R_ij = F_j(T_i) - F_i(T_i)
    
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
    # r = pmf.matmul(not_ec.transpose(0, 1))
    r = pmf.cumsum(1).matmul(y.transpose(0, 1))
    diag_r = r.diag().view(1, -1)
    r = ones.matmul(diag_r) - r
    return r.transpose(0, 1)

@numba.jit(nopython=True)
def _pair_rank_mat(mat, durations, events, dtype='float32'):
    n = len(durations)
    # mat = np.zeros((n, n), dtype=dtype)
    for i in range(n):
    # for i in numba.prange(n):
        dur_i = durations[i]
        ev_i = events[i]
        if ev_i == 0:
            continue
        for j in range(n):
            dur_j = durations[j]
            ev_j = events[j]
            if (dur_i < dur_j) or ((dur_i == dur_j) and (ev_j == 0)):
                mat[i, j] = 1
    return mat

def pair_rank_mat(durations, events, dtype='float32'):
    """Indicator matrix R with R_ij = 1{T_i < T_j and D_i = 1}.
    So it takes value 1 if we observe that i has an event before j and zero otherwise.
    
    Arguments:
        durations {np.array} -- Array with durations.
        events {np.array} -- Array with event indicators.
    
    Keyword Arguments:
        dtype {str} -- dtype of array (default: {'float32'})
    
    Returns:
        np.array -- n x n matrix indicating if i has an observerd event before j.
    """
    durations = durations.reshape(-1)
    events = events.reshape(-1)
    n = len(durations)
    mat = np.zeros((n, n), dtype=dtype)
    mat = _pair_rank_mat(mat, durations, events, dtype)
    return mat

# @numba.jit(nopython=True)
# def _not_e_or_c_mat(mat, y, dtype):
#     rows, cols = y.shape
#     for i in range(rows):
#         for j in range(cols):
#             mat[i, j] = 1
#             if y[i, j] == 1:
#                 break
#     return mat

# def not_e_or_c_mat(y, dtype='float32'):
#     """Makes a copy of y with ones to the left of the single indicator in y."""
#     mat = np.zeros_like(y)
#     mat = _not_e_or_c_mat(mat, y, dtype)
#     return mat

class DatasetDeepHitSingle(DatasetTuple):
    def __getitem__(self, index):
        input, target =  super().__getitem__(index).to_numpy()
        y, events = target.to_numpy()
        durations = np.nonzero(y)[1]
        rank_mat = pair_rank_mat(durations, events)
        # not_e_c = not_e_or_c_mat(y)
        # target = tuplefy(*target, not_e_c, rank_mat)
        target = tuplefy(*target, rank_mat)
        return tuplefy(input, target).to_tensor()

# def loss_deephit_l2(phi, y, d, sigma=1., reduction='elementwise_mean', _epsilon=1e-7):
#     idx_sort = y.nonzero()[:, 1].sort()[1]
#     phi, y, d = phi[idx_sort], y[idx_sort], d[idx_sort]
#     cif = phi.softmax(1).cumsum(1)
#     # f_si = phi.mul(y).sum(1)
#     loss = torch.zeros_like(d)
#     _acc = torch.zeros_like(d)
#     _etas = torch.zeros_like(d)
#     for row, idx in y[:-1].nonzero():
# #         acceptable = y[(row+1):, idx].eq(0.).float().mul(d[(row+1):]).mul(d[row])
#         acceptable = y[(row+1):, idx].mul(d[(row+1):]).eq(0.).float().mul(d[row])
#         if acceptable.sum() == 0:
#             loss[row] = 0.
#             continue
#         eta = (-cif[row, idx].sub(cif[(row+1):, idx]).div(sigma)).exp()
# #         print(eta)
#         loss[row] = eta.mul(acceptable).sum() / acceptable.sum()
#         _acc[row] = acceptable.sum()
#         _etas[row] = eta.mul(acceptable).sum()
# #     print('end')
#     if reduction == 'none':
#         return loss#, _acc, _etas
#     elif reduction == 'elementwise_mean':
# #         return loss.mean()
#         n = loss.nonzero().shape[0]
#         loss = loss.sum()
#         if n != 0:
#             loss = loss / n
#         return loss
#     return loss.sum()

