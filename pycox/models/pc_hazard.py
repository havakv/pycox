import warnings
import pandas as pd
import torch
import torch.nn.functional as F
import torchtuples as tt
from pycox import models
from pycox.models.utils import array_or_tensor, pad_col, make_subgrid
from pycox.preprocessing import label_transforms

class PCHazard(models.base._SurvModelBase):
    """Continuous hazard parametrization of the survival likelihood.

    NOTE:
        The label_transform is slightly different than regular hazard transform in terms
        of censorings beeing moved right instead of left.
        Also, this typically has one less output node than the HazardSurv.
    """
    label_transform = label_transforms.LabTransPCHazard

    def __init__(self, net, optimizer=None, device=None, duration_index=None, sub=1):
        self.duration_index = duration_index
        self.sub = sub
        super().__init__(net, self.make_loss(), optimizer, device)
        if self.duration_index is not None:
            self._check_out_features()

    def make_loss(self):
        return models.loss.NLLPCHazardLoss()

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
        surv = hazard.cumsum(1).mul(-1).exp().transpose(0, 1)
        return array_or_tensor(surv, numpy, input)

    def predict_hazard(self, input, batch_size=8224, numpy=None, eval_=True, to_cpu=False, num_workers=0):
        """Predict the hazard function for `input`.

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
            [np.ndarray or tensor] -- Predicted hazards
        """
        preds = self.predict(input, batch_size, False, eval_, False, to_cpu, num_workers)
        n = preds.shape[0]
        hazard = F.softplus(preds).view(-1, 1).repeat(1, self.sub).view(n, -1).div(self.sub)
        hazard = pad_col(hazard, where='start')
        return array_or_tensor(hazard, numpy, input)

    def predict_surv_df(self, input, batch_size=8224, eval_=True, num_workers=0):
        self._check_out_features()
        surv = self.predict_surv(input, batch_size, True, eval_, True, num_workers)
        index = None
        if self.duration_index is not None:
            index = make_subgrid(self.duration_index, self.sub)
        return pd.DataFrame(surv, index)

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
                    raise ValueError(f"Ouput of `net` is {m_output}, but data only trains {max_idx + 1} indices. "+
                        f"Output of `net` should be  {max_idx + 1}."+
                        "Set `check_out_feature=False` to suppress this Error.")
