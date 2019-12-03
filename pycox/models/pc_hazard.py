import warnings
import pandas as pd
import torch
import torch.nn.functional as F
import torchtuples as tt
from pycox import models
from pycox.models.utils import pad_col, make_subgrid
from pycox.preprocessing import label_transforms

class PCHazard(models.base.SurvBase):
    """The PC-Hazard (piecewise constant hazard) method from [1].
    The Piecewise Constant Hazard (PC-Hazard) model from [1] which assumes that the continuous-time
    hazard function is constant in a set of predefined intervals. It is similar to the Piecewise
    Exponential Models [2] but with a softplus activation instead of the exponential function.

    Note that the label_transform is slightly different than that of the LogistcHazard and PMF methods.
    This typically results in one less output node.

    References:
    [1] Håvard Kvamme and Ørnulf Borgan. Continuous and Discrete-Time Survival Prediction
        with Neural Networks. arXiv preprint arXiv:1910.06724, 2019.
        https://arxiv.org/pdf/1910.06724.pdf

    [2] Michael Friedman. Piecewise exponential models for survival data with covariates.
        The Annals of Statistics, 10(1):101–113, 1982.
        https://projecteuclid.org/euclid.aos/1176345693
    """
    label_transform = label_transforms.LabTransPCHazard

    def __init__(self, net, optimizer=None, device=None, duration_index=None, sub=1, loss=None):
        self.duration_index = duration_index
        self.sub = sub
        if loss is None:
            loss = models.loss.NLLPCHazardLoss()
        super().__init__(net, loss, optimizer, device)
        if self.duration_index is not None:
            self._check_out_features()

    @property
    def sub(self):
        return self._sub

    @sub.setter
    def sub(self, sub):
        if type(sub) is not int:
            raise ValueError(f"Need `sub` to have type `int`, got {type(sub)}")
        self._sub = sub

    def predict_surv(self, input, batch_size=8224, numpy=None, eval_=True, to_cpu=False, num_workers=0):
        hazard = self.predict_hazard(input, batch_size, False, eval_, to_cpu, num_workers)
        surv = hazard.cumsum(1).mul(-1).exp()
        return tt.utils.array_or_tensor(surv, numpy, input)

    def predict_hazard(self, input, batch_size=8224, numpy=None, eval_=True, to_cpu=False, num_workers=0):
        """Predict the hazard function for `input`.

        Arguments:
            input {tuple, np.ndarra, or torch.tensor} -- Input to net.
        
        Keyword Arguments:
            batch_size {int} -- Batch size (default: {8224})
            numpy {bool} -- 'False' gives tensor, 'True' gives numpy, and None give same as input
                (default: {None})
            eval_ {bool} -- If 'True', use 'eval' mode on net. (default: {True})
            to_cpu {bool} -- For larger data sets we need to move the results to cpu
                (default: {False})
            num_workers {int} -- Number of workers in created dataloader (default: {0})
        
        Returns:
            [np.ndarray or tensor] -- Predicted hazards
        """
        preds = self.predict(input, batch_size, False, eval_, False, to_cpu, num_workers)
        n = preds.shape[0]
        hazard = F.softplus(preds).view(-1, 1).repeat(1, self.sub).view(n, -1).div(self.sub)
        hazard = pad_col(hazard, where='start')
        return tt.utils.array_or_tensor(hazard, numpy, input)

    def predict_surv_df(self, input, batch_size=8224, eval_=True, num_workers=0):
        self._check_out_features()
        surv = self.predict_surv(input, batch_size, True, eval_, True, num_workers)
        index = None
        if self.duration_index is not None:
            index = make_subgrid(self.duration_index, self.sub)
        return pd.DataFrame(surv.transpose(), index)

    def fit(self, input, target, batch_size=256, epochs=1, callbacks=None, verbose=True,
            num_workers=0, shuffle=True, metrics=None, val_data=None, val_batch_size=8224,
            check_out_features=True, **kwargs):
        if check_out_features:
            self._check_out_features(target)
        return super().fit(input, target, batch_size, epochs, callbacks, verbose, num_workers,
                           shuffle, metrics, val_data, val_batch_size, **kwargs)

    def fit_dataloader(self, dataloader, epochs=1, callbacks=None, verbose=True, metrics=None,
                       val_dataloader=None, check_out_features=True):
        if check_out_features:
            self._check_out_features()
        return super().fit_dataloader(dataloader, epochs, callbacks, verbose, metrics, val_dataloader)

    def _check_out_features(self, target=None):
        last = list(self.net.modules())[-1]
        if hasattr(last, 'out_features'):
            m_output = last.out_features
            if self.duration_index is not None:
                n_grid = len(self.duration_index)
                if n_grid == m_output:
                    raise ValueError("Output of `net` is one too large. Should have length "+
                        f"{len(self.duration_index)-1}")
                if n_grid != (m_output + 1):
                    raise ValueError(f"Output of `net` does not correspond with `duration_index`")
            if target is not None:
                max_idx = tt.tuplefy(target).to_numpy()[0].max()
                if m_output != (max_idx + 1):
                    raise ValueError(f"Output of `net` is {m_output}, but data only trains {max_idx + 1} indices. "+
                        f"Output of `net` should be  {max_idx + 1}."+
                        "Set `check_out_feature=False` to suppress this Error.")
