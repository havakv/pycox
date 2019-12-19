import numpy as np
import pandas as pd
import torch
import torchtuples as tt

from pycox import models
from pycox.models.utils import pad_col

class DeepHitSingle(models.pmf.PMFBase):
    """The DeepHit methods by [1] but only for single event (not competing risks).

    Note that `alpha` is here defined differently than in [1], as `alpha` is  weighting between
    the likelihood and rank loss (see Appendix D in [2])
        loss = alpha * nll + (1 - alpha) rank_loss(sigma).
    
    Also, unlike [1], this implementation allows for survival past the max durations, i.e., it
    does not assume all events happen within the defined duration grid. See [3] for details.
    
    Keyword Arguments:
        alpha {float} -- Weighting (0, 1) likelihood and rank loss (L2 in paper).
            1 gives only likelihood, and 0 gives only rank loss. (default: {0.2})
        sigma {float} -- from eta in rank loss (L2 in paper) (default: {0.1})
    
    References:
    [1] Changhee Lee, William R Zame, Jinsung Yoon, and Mihaela van der Schaar. Deephit: A deep learning
        approach to survival analysis with competing risks. In Thirty-Second AAAI Conference on Artificial
        Intelligence, 2018.
        http://medianetlab.ee.ucla.edu/papers/AAAI_2018_DeepHit

    [2] Håvard Kvamme, Ørnulf Borgan, and Ida Scheel.
        Time-to-event prediction with neural networks and Cox regression.
        Journal of Machine Learning Research, 20(129):1–30, 2019.
        http://jmlr.org/papers/v20/18-424.html
    
    [3] Håvard Kvamme and Ørnulf Borgan. Continuous and Discrete-Time Survival Prediction
        with Neural Networks. arXiv preprint arXiv:1910.06724, 2019.
        https://arxiv.org/pdf/1910.06724.pdf
    """
    def __init__(self, net, optimizer=None, device=None, duration_index=None, alpha=0.2, sigma=0.1, loss=None):
        if loss is None:
            loss = models.loss.DeepHitSingleLoss(alpha, sigma)
        super().__init__(net, loss, optimizer, device, duration_index)

    def make_dataloader(self, data, batch_size, shuffle, num_workers=0):
        dataloader = super().make_dataloader(data, batch_size, shuffle, num_workers,
                                             make_dataset=models.data.DeepHitDataset)
        return dataloader
    
    def make_dataloader_predict(self, input, batch_size, shuffle=False, num_workers=0):
        dataloader = super().make_dataloader(input, batch_size, shuffle, num_workers)
        return dataloader


class DeepHit(tt.Model):
    """DeepHit for competing risks [1].
    For single risk (only one event type) use `DeepHitSingle` instead!

    Note that `alpha` is here defined differently than in [1], as `alpha` is  weighting between
    the likelihood and rank loss (see Appendix D in [2])
        loss = alpha * nll + (1 - alpha) rank_loss(sigma).

    Also, unlike [1], this implementation allows for survival past the max durations, i.e., it
    does not assume all events happen within the defined duration grid. See [3] for details.
    
    Keyword Arguments:
        alpha {float} -- Weighting (0, 1) likelihood and rank loss (L2 in paper).
            1 gives only likelihood, and 0 gives only rank loss. (default: {0.2})
        sigma {float} -- from eta in rank loss (L2 in paper) (default: {0.1})

    References:
    [1] Changhee Lee, William R Zame, Jinsung Yoon, and Mihaela van der Schaar. Deephit: A deep learning
        approach to survival analysis with competing risks. In Thirty-Second AAAI Conference on Artificial
        Intelligence, 2018.
        http://medianetlab.ee.ucla.edu/papers/AAAI_2018_DeepHit

    [2] Håvard Kvamme, Ørnulf Borgan, and Ida Scheel.
        Time-to-event prediction with neural networks and Cox regression.
        Journal of Machine Learning Research, 20(129):1–30, 2019.
        http://jmlr.org/papers/v20/18-424.html
    
    [3] Håvard Kvamme and Ørnulf Borgan. Continuous and Discrete-Time Survival Prediction
        with Neural Networks. arXiv preprint arXiv:1910.06724, 2019.
        https://arxiv.org/pdf/1910.06724.pdf
    """
    def __init__(self, net, optimizer=None, device=None, alpha=0.2, sigma=0.1, duration_index=None, loss=None):
        self.duration_index = duration_index
        if loss is None:
            loss = models.loss.DeepHitLoss(alpha, sigma)
        super().__init__(net, loss, optimizer, device)

    @property
    def duration_index(self):
        """
        Array of durations that defines the discrete times. This is used to set the index
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
                                             make_dataset=models.data.DeepHitDataset)
        return dataloader
    
    def make_dataloader_predict(self, input, batch_size, shuffle=False, num_workers=0):
        dataloader = super().make_dataloader(input, batch_size, shuffle, num_workers)
        return dataloader

    def predict_surv_df(self, input, batch_size=8224, eval_=True, num_workers=0):
        """Predict the survival function for `input`, i.e., survive all of the event types,
        and return as a pandas DataFrame.
        See `prediction_surv_df` to return a DataFrame instead.

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
        """Predict the survival function for `input`, i.e., survive all of the event types.
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
        return tt.utils.array_or_tensor(surv, numpy, input)

    def predict_cif(self, input, batch_size=8224, numpy=None, eval_=True,
                     to_cpu=False, num_workers=0):
        """Predict the cumulative incidence function (cif) for `input`.

        Arguments:
            input {tuple, np.ndarray, or torch.tensor} -- Input to net.
        
        Keyword Arguments:
            batch_size {int} -- Batch size (default: {8224})
            numpy {bool} -- 'False' gives tensor, 'True' gives numpy, and None give same as input
                (default: {None})
            eval_ {bool} -- If 'True', use 'eval' mode on net. (default: {True})
            to_cpu {bool} -- For larger data sets we need to move the results to cpu
                (default: {False})
            num_workers {int} -- Number of workers in created dataloader (default: {0})
        
        Returns:
            [np.ndarray or tensor] -- Predictions
        """
        pmf = self.predict_pmf(input, batch_size, False, eval_, to_cpu, num_workers)
        cif = pmf.cumsum(1)
        return tt.utils.array_or_tensor(cif, numpy, input)

    def predict_pmf(self, input, batch_size=8224, numpy=None, eval_=True,
                     to_cpu=False, num_workers=0):
        """Predict the probability mass fuction (PMF) for `input`.

        Arguments:
            input {tuple, np.ndarray, or torch.tensor} -- Input to net.
        
        Keyword Arguments:
            batch_size {int} -- Batch size (default: {8224})
            numpy {bool} -- 'False' gives tensor, 'True' gives numpy, and None give same as input
                (default: {None})
            eval_ {bool} -- If 'True', use 'eval' mode on net. (default: {True})
            grads {bool} -- If gradients should be computed (default: {False})
            to_cpu {bool} -- For larger data sets we need to move the results to cpu
                (default: {False})
            num_workers {int} -- Number of workers in created dataloader (default: {0})
        
        Returns:
            [np.ndarray or tensor] -- Predictions
        """
        preds = self.predict(input, batch_size, False, eval_, False, to_cpu, num_workers)
        pmf = pad_col(preds.view(preds.size(0), -1)).softmax(1)[:, :-1]
        pmf = pmf.view(preds.shape).transpose(0, 1).transpose(1, 2)
        return tt.utils.array_or_tensor(pmf, numpy, input)
