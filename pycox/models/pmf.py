import pandas as pd
import torchtuples as tt
from pycox import models
from pycox.models.utils import pad_col
from pycox.preprocessing import label_transforms
from pycox.models.interpolation import InterpolatePMF


class PMFBase(models.base.SurvBase):
    """Base class for PMF methods.
    """
    label_transform = label_transforms.LabTransDiscreteTime

    def __init__(self, net, loss=None, optimizer=None, device=None, duration_index=None):
        self.duration_index = duration_index
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

    def predict_surv(self, input, batch_size=8224, numpy=None, eval_=True, to_cpu=False,
                     num_workers=0):
        pmf = self.predict_pmf(input, batch_size, False, eval_, to_cpu, num_workers)
        surv = 1 - pmf.cumsum(1)
        return tt.utils.array_or_tensor(surv, numpy, input)

    def predict_pmf(self, input, batch_size=8224, numpy=None, eval_=True, to_cpu=False,
                    num_workers=0):
        preds = self.predict(input, batch_size, False, eval_, False, to_cpu, num_workers)
        pmf = pad_col(preds).softmax(1)[:, :-1]
        return tt.utils.array_or_tensor(pmf, numpy, input)

    def predict_surv_df(self, input, batch_size=8224, eval_=True, num_workers=0):
        surv = self.predict_surv(input, batch_size, True, eval_, True, num_workers)
        return pd.DataFrame(surv.transpose(), self.duration_index)

    def interpolate(self, sub=10, scheme='const_pdf', duration_index=None):
        """Use interpolation for predictions.
        There are only one scheme:
            `const_pdf` and `lin_surv` which assumes pice-wise constant pmf in each interval (linear survival).
        
        Keyword Arguments:
            sub {int} -- Number of "sub" units in interpolation grid. If `sub` is 10 we have a grid with
                10 times the number of grid points than the original `duration_index` (default: {10}).
            scheme {str} -- Type of interpolation {'const_hazard', 'const_pdf'}.
                See `InterpolateDiscrete` (default: {'const_pdf'})
            duration_index {np.array} -- Cuts used for discretization. Does not affect interpolation,
                only for setting index in `predict_surv_df` (default: {None})
        
        Returns:
            [InterpolationPMF] -- Object for prediction with interpolation.
        """
        if duration_index is None:
            duration_index = self.duration_index
        return InterpolatePMF(self, scheme, duration_index, sub)


class PMF(PMFBase):
    """
    The PMF is a discrete-time survival model that parametrize the probability mass function (PMF)
    and optimizer the survival likelihood. It is the foundation of methods such as DeepHit and MTLR.
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

    References:
    [1] Håvard Kvamme and Ørnulf Borgan. Continuous and Discrete-Time Survival Prediction
        with Neural Networks. arXiv preprint arXiv:1910.06724, 2019.
        https://arxiv.org/pdf/1910.06724.pdf
    """
    def __init__(self, net, optimizer=None, device=None, duration_index=None, loss=None):
        if loss is None:
            loss = models.loss.NLLPMFLoss()
        super().__init__(net, loss, optimizer, device, duration_index)

