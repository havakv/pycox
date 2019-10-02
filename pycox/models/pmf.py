import pandas as pd
from pycox import models
from pycox.models.utils import array_or_tensor, pad_col


class _PMFBase(models.base._SurvModelBase):
    """Base class for PMF methods.
    """
    def __init__(self, net, optimizer=None, device=None, duration_index=None):
        self.duration_index = duration_index
        super().__init__(net, optimizer, device)

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

    def predict_surv(self, input, batch_size=8224, numpy=None, eval_=True, to_cpu=False,
                     num_workers=0):
        pmf = self.predict_pmf(input, batch_size, False, eval_, to_cpu, num_workers)
        surv = 1 - pmf.cumsum(0)
        return array_or_tensor(surv, numpy, input)

    def predict_pmf(self, input, batch_size=8224, numpy=None, eval_=True, to_cpu=False,
                    num_workers=0):
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
            [TupleTree, np.ndarray or tensor] -- Predictions
        """
        preds = self.predict(input, batch_size, False, eval_, False, to_cpu, num_workers)
        pmf = pad_col(preds).softmax(1)[:, :-1].transpose(0, 1)
        return array_or_tensor(pmf, numpy, input)

    def predict_surv_df(self, input, batch_size=8224, eval_=True, num_workers=0):
        surv = self.predict_surv(input, batch_size, True, eval_, True, num_workers)
        return pd.DataFrame(surv, self.duration_index)


class PMF(_PMFBase):
    """
    A discrete-time survival model that minimize the likelihood for right-censored data.
    See <link>.

    Arguments:
        net {torch.nn.Module} -- A torch module.
    
    Keyword Arguments:
        optimizer {Optimizer} -- A torch optimizer or similar. Preferrably use torchtuples.optim instead of
            torch.optim, as this allows for reinitialization, etc. If 'None' set to torchtuples.optim.AdamW.
            (default: {None})
        device {str, int, torch.device} -- Device to compute on. (default: {None})
            Preferrably pass a torch.device object.
            If 'None': use default gpu if avaiable, else use cpu.
            If 'int': used that gpu: torch.device('cuda:<device>').
            If 'string': string is passed to torch.device('string').
        duration_index {list, np.array} -- Array of durations that defineds the discrete times.
            This is used to set the index of the DataFrame in `predict_surv_df`.
    """
    @property
    def _loss(self):
        return models.loss.NLLPMFLoss()

