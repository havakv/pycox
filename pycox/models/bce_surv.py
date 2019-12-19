"""Estimate survival curve with binomial log-likelihood.

This method is not smart to use!!!!!!!
"""
import pandas as pd
import torch
from pycox import models
from pycox.preprocessing import label_transforms


class BCESurv(models.base.SurvBase):
    """
    The BCESurv method is a discrete-time survival model that parametrize the survival function directly
    and disregards individuals as they are censored. Each output node represents a binary classifier at 
    the corresponding time, where all censored individual are removed.
    See [1] for details.

    Arguments:
        net {torch.nn.Module} -- A torch module.

    Keyword Arguments:
        optimizer {Optimizer} -- A torch optimizer or similar. Preferably use torchtuples.optim instead of
            torch.optim, as this allows for reinitialization, etc. If 'None' set to torchtuples.optim.AdamW.
            (default: {None})
        device {str, int, torch.device} -- Device to compute on. (default: {None})
            Preferably pass a torch.device object.
            If 'None': use default gpu if available, else use cpu.
            If 'int': used that gpu: torch.device('cuda:<device>').
            If 'string': string is passed to torch.device('string').
        duration_index {list, np.array} -- Array of durations that defines the discrete times.
            This is used to set the index of the DataFrame in `predict_surv_df`.
        loss {func} -- An alternative loss function (default: {None})

    References:
        [1] Håvard Kvamme and Ørnulf Borgan. The Brier Score under Administrative Censoring: Problems
            and Solutions. arXiv preprint arXiv:1912.08581, 2019.
            https://arxiv.org/pdf/1912.08581.pdf
    """
    label_transform = label_transforms.LabTransDiscreteTime

    def __init__(self, net, optimizer=None, device=None, duration_index=None, loss=None):
        self.duration_index = duration_index
        if loss is None:
            loss = models.loss.BCESurvLoss()
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

    def predict_surv_df(self, input, batch_size=8224, eval_=True, num_workers=0, is_dataloader=None):
        surv = self.predict_surv(input, batch_size, True, eval_, True, num_workers, is_dataloader)
        return pd.DataFrame(surv.transpose(), self.duration_index)

    def predict_surv(self, input, batch_size=8224, numpy=None, eval_=True, to_cpu=False,
                     num_workers=0, is_dataloader=None):
        return self.predict(input, batch_size, numpy, eval_, False, to_cpu, num_workers,
                            is_dataloader, torch.sigmoid)

    def interpolate(self, sub=10, scheme='const_pdf', duration_index=None):
        """Use interpolation for predictions.
        There is only one scheme:
            `const_pdf` and `lin_surv` which assumes pice-wise constant PMF in each interval (linear survival).
        
        Keyword Arguments:
            sub {int} -- Number of "sub" units in interpolation grid. If `sub` is 10 we have a grid with
                10 times the number of grid points than the original `duration_index` (default: {10}).
            scheme {str} -- Type of interpolation {'const_pdf'}.
                See `InterpolateDiscrete` (default: {'const_pdf'})
            duration_index {np.array} -- Cuts used for discretization. Does not affect interpolation,
                only for setting index in `predict_surv_df` (default: {None})
        
        Returns:
            [InterpolateLogisticHazard] -- Object for prediction with interpolation.
        """
        if duration_index is None:
            duration_index = self.duration_index
        return models.interpolation.InterpolateDiscrete(self, scheme, duration_index, sub)
