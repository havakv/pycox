
import numpy as np
import pandas as pd
import torch
import torchtuples as tt
from pycox import models
from pycox.models.utils import array_or_tensor, pad_col, make_subgrid
from pycox.preprocessing import label_transforms

class LogisticHazard(models.base._SurvModelBase):
    """Essentailly same as pyth.Model, but instead of specifying a loss function,
    """
    label_transform = label_transforms.LabTransDiscreteTime

    def __init__(self, net, optimizer=None, device=None, duration_index=None):
        self.duration_index = duration_index
        super().__init__(net, self.make_loss(), optimizer, device)

    def make_loss(self):
        return models.loss.NLLLogistiHazardLoss()


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

    def predict_surv_df(self, input, batch_size=8224, eval_=True, num_workers=0):
        surv = self.predict_surv(input, batch_size, True, eval_, True, num_workers)
        return pd.DataFrame(surv, self.duration_index)

    def predict_surv(self, input, batch_size=8224, numpy=None, eval_=True, to_cpu=False,
                     num_workers=0, epsilon=1e-7):
        hazard = self.predict_hazard(input, batch_size, False, eval_, to_cpu, num_workers)
        surv = (1 - hazard).add(epsilon).log().cumsum(1).exp().transpose(0, 1)
        return array_or_tensor(surv, numpy, input)


    def predict_hazard(self, input, batch_size=8224, numpy=None, eval_=True, to_cpu=False,
                       num_workers=0):
        """Predict the hazard function for `input`.

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
            [np.ndarray or tensor] -- Predicted hazards
        """
        hazard = self.predict(input, batch_size, False, eval_, False, to_cpu, num_workers).sigmoid()
        return array_or_tensor(hazard, numpy, input)

    def interpolate(self,scheme='const_pdf', duration_index=None, sub=10):
        """Use interpolation for predictions.
        There are two schemes:
            `const_hazard` and `exp_surv` which assumes pice-wise constant hazard in each interval (exponential survival).
            `const_pdf` and `lin_surv` which assumes pice-wise constant pmf in each interval (linear survival).
        
        Keyword Arguments:
            scheme {str} -- Type of interpolation {'const_hazard', 'const_pdf'}.
                See `InterpolateDiscrete` (default: {'const_pdf'})
            duration_index {np.array} -- Cuts used for discretization. Does not affect interpolation,
                only for setting index in `predict_surv_df` (default: {None})
            sub {int} -- Number of "sub" units in interpolation grid. If `sub` is 10 we have a grid with
                10 times the number of grid points than the original `duration_index` (default: {10}).
        
        Returns:
            [InterpolationDiscrete] -- Object for prediction with interpolation.
        """
        if duration_index is None:
            duration_index = self.duration_index
        return InterpolateDiscrete(self, scheme, duration_index, sub)


class InterpolateDiscrete:
    """Interpolation of discrete models, for continuous predictions.
    There are two schemes:
        `const_hazard` and `exp_surv` which assumes pice-wise constant hazard in each interval (exponential survival).
        `const_pdf` and `lin_surv` which assumes pice-wise constant pmf in each interval (linear survival).
    
    Arguments:
        model {[type]} -- [description]

    Keyword Arguments:
        duration_index {np.array} -- Cuts used for discretization. Does not affect interpolation,
            only for setting index in `predict_surv_df` (default: {None})
        scheme {str} -- Type of interpolation {'const_hazard', 'const_pdf'} (default: {'const_pdf'})
        sub {int} -- Number of "sub" units in interpolation grid. If `sub` is 10 we have a grid with
            10 times the number of grid points than the original `duration_index` (default: {10}).
    
    Keyword Arguments:
    """
    def __init__(self, model, scheme='const_pdf', duration_index=None, sub=10, epsilon=1e-7):
        self.model = model
        self.scheme = scheme
        self.duration_index = duration_index
        self.sub = sub
        self.epsilon = epsilon

    @property
    def sub(self):
        return self._sub

    @sub.setter
    def sub(self, sub):
        if type(sub) is not int:
            raise ValueError(f"Need `sub` to have type `int`, got {type(sub)}")
        self._sub = sub

    def predict_hazard(self, input, batch_size=8224, numpy=None, eval_=True, to_cpu=False, num_workers=0):
        if self.scheme in ['const_hazard', 'exp_surv']:
            haz = self._hazard_const_haz(input, batch_size, numpy, eval_, to_cpu, num_workers)
        else:
            raise NotImplementedError
        return haz

    def predict_surv(self, input, batch_size=8224, numpy=None, eval_=True, to_cpu=False, num_workers=0):
        if self.scheme in ['const_hazard', 'exp_surv']:
            surv = self._surv_const_haz(input, batch_size, numpy, eval_, to_cpu, num_workers)
        elif self.scheme in ['const_pdf', 'lin_surv']:
            surv = self._surv_const_pdf(input, batch_size, numpy, eval_, to_cpu, num_workers)
        else:
            raise NotImplementedError
        return surv

    def _surv_const_pdf(self, input, batch_size=8224, numpy=None, eval_=True, to_cpu=False,
                        num_workers=0):
        s = self.model.predict_surv(input, batch_size, False, eval_, to_cpu, num_workers, self.epsilon)
        s = s.transpose(0, 1)
        n, m = s.shape
        diff = (s[:, 1:] - s[:, :-1]).contiguous().view(-1, 1).repeat(1, self.sub).view(n, -1)
        rho = torch.linspace(0, 1, self.sub+1)[:-1].contiguous().repeat(n, m-1)
        s_prev = s[:, :-1].contiguous().view(-1, 1).repeat(1, self.sub).view(n, -1)
        surv = torch.zeros(n, int((m-1)*self.sub + 1))
        surv[:, :-1] = diff * rho + s_prev
        surv[:, -1] = s[:, -1]
        surv = surv.transpose(0, 1)
        return array_or_tensor(surv, numpy, input)

    def _hazard_const_haz(self, input, batch_size=8224, numpy=None, eval_=True, to_cpu=False,
                          num_workers=0):
        haz = self.model.predict_hazard(input, batch_size, False, eval_, to_cpu, num_workers)
        haz = (1 - haz).add(self.epsilon).log().mul(-1).relu()[:, 1:].contiguous()
        n = haz.shape[0]
        haz = haz.view(-1, 1).repeat(1, self.sub).view(n, -1).div(self.sub)
        haz = pad_col(haz, where='start')
        return array_or_tensor(haz, numpy, input)

    def _surv_const_haz(self, input, batch_size=8224, numpy=None, eval_=True, to_cpu=False,
                        num_workers=0):
        haz = self._hazard_const_haz(input, self.sub, batch_size, False, eval_, to_cpu, num_workers)
        surv = haz.cumsum(1).mul(-1).exp().transpose(0, 1)
        return array_or_tensor(surv, numpy, input)

    def predict_surv_df(self, input, batch_size=8224, eval_=True, num_workers=0):
        surv = self.predict_surv(input, self.sub, batch_size, True, eval_, num_workers)
        index = None
        if self.duration_index is not None:
            index = make_subgrid(self.duration_index, self.sub)
        return pd.DataFrame(surv, index)
