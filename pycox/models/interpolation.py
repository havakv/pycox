import pandas as pd
import torch
import torchtuples as tt
from pycox.models import utils


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

    @property
    def sub(self):
        return self._sub

    @sub.setter
    def sub(self, sub):
        if type(sub) is not int:
            raise ValueError(f"Need `sub` to have type `int`, got {type(sub)}")
        self._sub = sub

    def predict_hazard(self, input, batch_size=8224, numpy=None, eval_=True, to_cpu=False, num_workers=0):
        raise NotImplementedError

    def predict_pmf(self, input, batch_size=8224, numpy=None, eval_=True, to_cpu=False, num_workers=0):
        raise NotImplementedError

    def predict_surv(self, input, batch_size=8224, numpy=None, eval_=True, to_cpu=False, num_workers=0):
        """Predict the survival function for `input`.
        See `prediction_surv_df` to return a DataFrame instead.

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
        return self._surv_const_pdf(input, batch_size, numpy, eval_, to_cpu, num_workers)

    def _surv_const_pdf(self, input, batch_size=8224, numpy=None, eval_=True, to_cpu=False,
                        num_workers=0):
        """Basic method for constant PDF interpolation that use `self.model.predict_surv`.

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
        s = self.model.predict_surv(input, batch_size, False, eval_, to_cpu, num_workers)
        n, m = s.shape
        device = s.device
        diff = (s[:, 1:] - s[:, :-1]).contiguous().view(-1, 1).repeat(1, self.sub).view(n, -1)
        rho = torch.linspace(0, 1, self.sub+1, device=device)[:-1].contiguous().repeat(n, m-1)
        s_prev = s[:, :-1].contiguous().view(-1, 1).repeat(1, self.sub).view(n, -1)
        surv = torch.zeros(n, int((m-1)*self.sub + 1))
        surv[:, :-1] = diff * rho + s_prev
        surv[:, -1] = s[:, -1]
        return tt.utils.array_or_tensor(surv, numpy, input)

    def predict_surv_df(self, input, batch_size=8224, eval_=True, to_cpu=False, num_workers=0):
        """Predict the survival function for `input` and return as a pandas DataFrame.
        See `predict_surv` to return tensor or np.array instead.

        Arguments:
            input {tuple, np.ndarray, or torch.tensor} -- Input to net.
        
        Keyword Arguments:
            batch_size {int} -- Batch size (default: {8224})
            eval_ {bool} -- If 'True', use 'eval' mode on net. (default: {True})
            num_workers {int} -- Number of workers in created dataloader (default: {0})
        
        Returns:
            pd.DataFrame -- Predictions
        """
        surv = self.predict_surv(input, batch_size, True, eval_, to_cpu, num_workers)
        index = None
        if self.duration_index is not None:
            index = utils.make_subgrid(self.duration_index, self.sub)
        return pd.DataFrame(surv.transpose(), index)


class InterpolatePMF(InterpolateDiscrete):
    def predict_pmf(self, input, batch_size=8224, numpy=None, eval_=True, to_cpu=False, num_workers=0):
        if not self.scheme in ['const_pdf', 'lin_surv']:
            raise NotImplementedError
        pmf = self.model.predict_pmf(input, batch_size, False, eval_, to_cpu, num_workers)
        n, m = pmf.shape
        pmf_cdi = pmf[:, 1:].contiguous().view(-1, 1).repeat(1, self.sub).div(self.sub).view(n, -1)
        pmf_cdi = utils.pad_col(pmf_cdi, where='start')
        pmf_cdi[:, 0] = pmf[:, 0]
        return tt.utils.array_or_tensor(pmf_cdi, numpy, input)

    def _surv_const_pdf(self, input, batch_size=8224, numpy=None, eval_=True, to_cpu=False, num_workers=0):
        pmf = self.predict_pmf(input, batch_size, False, eval_, to_cpu, num_workers)
        surv = 1 - pmf.cumsum(1)
        return tt.utils.array_or_tensor(surv, numpy, input)


class InterpolateLogisticHazard(InterpolateDiscrete):
    epsilon = 1e-7
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

    def _hazard_const_haz(self, input, batch_size=8224, numpy=None, eval_=True, to_cpu=False,
                          num_workers=0):
        """Computes the continuous-time constant hazard interpolation.
        Essentially we what the discrete survival estimates to match the continuous time at the knots.
        So essentially we want
            $$S(tau_j) = prod_{k=1}^j [1 - h_k] = prod_{k=1}{j} exp[-eta_k].$$
        where $h_k$ is the discrete hazard estimates and $eta_k$ continuous time hazards multiplied
        with the length of the duration interval as they are defined for the PC-Hazard method.
        Thus we get 
            $$eta_k = - log[1 - h_k]$$
        which can be divided by the length of the time interval to get the continuous time hazards.
        """
        haz_orig = self.model.predict_hazard(input, batch_size, False, eval_, to_cpu, num_workers)
        haz = (1 - haz_orig).add(self.epsilon).log().mul(-1).relu()[:, 1:].contiguous()
        n = haz.shape[0]
        haz = haz.view(-1, 1).repeat(1, self.sub).view(n, -1).div(self.sub)
        haz = utils.pad_col(haz, where='start')
        haz[:, 0] = haz_orig[:, 0]
        return tt.utils.array_or_tensor(haz, numpy, input)

    def _surv_const_haz(self, input, batch_size=8224, numpy=None, eval_=True, to_cpu=False, num_workers=0):
        haz = self._hazard_const_haz(input, batch_size, False, eval_, to_cpu, num_workers)
        surv_0 = 1 - haz[:, :1]
        surv = utils.pad_col(haz[:, 1:], where='start').cumsum(1).mul(-1).exp().mul(surv_0)
        return tt.utils.array_or_tensor(surv, numpy, input)
