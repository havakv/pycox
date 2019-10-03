"""Impleentation of the deephit model, but currently for single event cause
(not compteeting risk).
"""
import numpy as np
import pandas as pd
import numba
import torch
import torchtuples as tt

from pycox import models
from pycox.models.utils import array_or_tensor, pad_col

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


class DeepHitSingle(models.pmf._PMFBase):
    """Essentailly same as torchtuples.Model, but instead of specifying a loss function,
    we now specify 'alpha' and 'simga'.
    See paper (or make_loss_deephit function)
    
    Keyword Arguments:
        alpha {float} -- Weighting (0, 1) likelihood and rank loss (L2 in paper).
            1 gives only likelihood, and 0 gives only rank loss. (default: {0.2})
        sigma {float} -- from eta in rank loss (L2 in paper) (default: {0.1})
    """
    def __init__(self, net, optimizer=None, device=None, duration_index=None, alpha=0.2, sigma=0.1):
        loss = self.make_loss(alpha, sigma)
        super().__init__(net, loss, optimizer, device, duration_index)

    def make_loss(self, alpha, sigma):
        return models.loss.DeepHitSingleLoss(alpha, sigma)

    def make_dataloader(self, data, batch_size, shuffle, num_workers=0):
        dataloader = super().make_dataloader(data, batch_size, shuffle, num_workers,
                                             make_dataset=DatasetDeepHit)
        return dataloader
    
    def make_dataloader_predict(self, input, batch_size, shuffle=False, num_workers=0):
        dataloader = super().make_dataloader(input, batch_size, shuffle, num_workers)
        return dataloader


class DatasetDeepHit(tt.data.DatasetTuple):
    def __getitem__(self, index):
        input, target =  super().__getitem__(index)
        target = target.to_numpy()
        rank_mat = pair_rank_mat(*target)
        target = tt.tuplefy(*target, rank_mat).to_tensor()
        return tt.tuplefy(input, target)


class DeepHit(tt.Model):
    """DeepHit for competing risks.
    If you have only one event type, use DeepHitSingle instead!

    Essentailly same as torchtuples.Model, but instead of specifying a loss function,
    we now specify 'alpha' and 'simga'.
    See paper (or make_loss_deephit function)
    
    Keyword Arguments:
        alpha {float} -- Weighting (0, 1) likelihood and rank loss (L2 in paper).
            1 gives only likelihood, and 0 gives only rank loss. (default: {0.2})
        sigma {float} -- from eta in rank loss (L2 in paper) (default: {0.1})
    """
    def __init__(self, net, optimizer=None, device=None, alpha=0.2, sigma=0.1, duration_index=None):
        self.duration_index = duration_index
        loss = self.make_loss(alpha, sigma)
        super().__init__(net, loss, optimizer, device)

    def make_loss(self, alpha, sigma):
        return models.loss.DeepHitLoss(alpha, sigma)

    @property
    def duration_index(self):
        """
        Array of durations that defineds the discrete times. This is used to set the index
        of the DataFrame in `predict_surv_df`.
        
        Returns:
            np.array -- Duration index.
        """
        return self._duration_index

    @duration_index.setter
    def duration_index(self, val):
        self._duration_index = val

    def make_dataloader(self, data, batch_size, shuffle, num_workers=0):
        dataloader = super().make_dataloader(data, batch_size, shuffle, num_workers,
                                             make_dataset=DatasetDeepHit)
        return dataloader
    
    def make_dataloader_predict(self, input, batch_size, shuffle=False, num_workers=0):
        dataloader = super().make_dataloader(input, batch_size, shuffle, num_workers)
        return dataloader

    def predict_surv_df(self, input, batch_size=8224, eval_=True, num_workers=0):
        """Predict the survival fuction for `input` and return as a pandas DataFrame.
        See `predict_surv` to return tensor or np.array instead.

        Arguments:
            input {tuple, np.ndarra, or torch.tensor} -- Input to net.
        
        Keyword Arguments:
            batch_size {int} -- Batch size (default: {8224})
            eval_ {bool} -- If 'True', use 'eval' modede on net. (default: {True})
            num_workers {int} -- Number of workes in created dataloader (default: {0})
        
        Returns:
            pd.DataFrame -- Predictions
        """
        surv = self.predict_surv(input, batch_size, True, eval_, True, num_workers)
        return pd.DataFrame(surv, self.duration_index)

    def predict_surv(self, input, batch_size=8224, numpy=None, eval_=True,
                     to_cpu=False, num_workers=0):
        """Predict the survival fuction for `input`, i.e., survive all of the event types.
        See `prediction_surv_df` to return a DataFrame instead.

        Arguments:
            input {tuple, np.ndarra, or torch.tensor} -- Input to net.
        
        Keyword Arguments:
            batch_size {int} -- Batch size (default: {8224})
            numpy {bool} -- 'False' gives tensor, 'True' gives numpy, and None give same as input
                (default: {None})
            eval_ {bool} -- If 'True', use 'eval' modede on net. (default: {True})
            to_cpu {bool} -- For larger data sets we need to move the results to cpu
                (default: {False})
            num_workers {int} -- Number of workes in created dataloader (default: {0})
        
        Returns:
            [TupleTree, np.ndarray or tensor] -- Predictions
        """
        cif = self.predict_cif(input, batch_size, False, eval_, to_cpu, num_workers)
        surv = 1. - cif.sum(0)
        return array_or_tensor(surv, numpy, input)

    def predict_cif(self, input, batch_size=8224, numpy=None, eval_=True,
                     to_cpu=False, num_workers=0):
        """Predict the cumulative incidence function (cif) for `input`.

        Arguments:
            input {tuple, np.ndarra, or torch.tensor} -- Input to net.
        
        Keyword Arguments:
            batch_size {int} -- Batch size (default: {8224})
            numpy {bool} -- 'False' gives tensor, 'True' gives numpy, and None give same as input
                (default: {None})
            eval_ {bool} -- If 'True', use 'eval' modede on net. (default: {True})
            to_cpu {bool} -- For larger data sets we need to move the results to cpu
                (default: {False})
            num_workers {int} -- Number of workes in created dataloader (default: {0})
        
        Returns:
            [np.ndarray or tensor] -- Predictions
        """
        pmf = self.predict_pmf(input, batch_size, False, eval_, to_cpu, num_workers)
        cif = pmf.cumsum(1)
        return array_or_tensor(cif, numpy, input)

    def predict_pmf(self, input, batch_size=8224, numpy=None, eval_=True,
                     to_cpu=False, num_workers=0):
        """Predict the probability mass fuction (PMF) for `input`.

        Arguments:
            input {tuple, np.ndarra, or torch.tensor} -- Input to net.
        
        Keyword Arguments:
            batch_size {int} -- Batch size (default: {8224})
            numpy {bool} -- 'False' gives tensor, 'True' gives numpy, and None give same as input
                (default: {None})
            eval_ {bool} -- If 'True', use 'eval' modede on net. (default: {True})
            grads {bool} -- If gradients should be computed (default: {False})
            to_cpu {bool} -- For larger data sets we need to move the results to cpu
                (default: {False})
            num_workers {int} -- Number of workes in created dataloader (default: {0})
        
        Returns:
            [np.ndarray or tensor] -- Predictions
        """
        preds = self.predict(input, batch_size, False, eval_, False, to_cpu, num_workers)
        pmf = pad_col(preds.view(preds.size(0), -1)).softmax(1)[:, :-1]
        pmf = pmf.view(preds.shape).transpose(0, 1).transpose(1, 2)
        return array_or_tensor(pmf, numpy, input)
