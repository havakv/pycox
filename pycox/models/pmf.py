from typing import Any, Callable, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch import Tensor
import torchtuples as tt

import pycox.models as models
from pycox.models.discrete_time import output2pmf, pmf2surv
from pycox.models.interpolation import InterpolatePMF
from pycox.models.loss import NLLPMFLoss
from pycox.preprocessing.label_transforms import LabTransDiscreteTime


class PMFBase(models.base.SurvBase):
    """Base class for PMF methods.
    """

    label_transform = LabTransDiscreteTime

    def __init__(
        self,
        net: torch.nn.Module,
        loss: Callable,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: Optional[Union[str, int, torch.device]] = None,
        duration_index: Optional[np.ndarray] = None,
    ) -> None:
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

    def predict_surv_df(
        self, input: Any, batch_size: int = 8224, eval_: bool = True, num_workers: int = 0,
    ) -> pd.DataFrame:
        surv = self.predict_surv(input, batch_size, True, eval_, True, num_workers)
        return pd.DataFrame(surv.transpose(), self.duration_index)

    def predict_surv(
        self,
        input: Any,
        batch_size: int = 8224,
        numpy: Optional[bool] = None,
        eval_: bool = True,
        to_cpu: bool = False,
        num_workers: int = 0,
    ) -> Union[Tensor, np.ndarray]:
        pmf = self.predict_pmf(input, batch_size, False, eval_, to_cpu, num_workers)
        surv = pmf2surv(pmf)
        return tt.utils.array_or_tensor(surv, numpy, input)

    def predict_pmf(
        self,
        input: Any,
        batch_size: int = 8224,
        numpy: Optional[bool] = None,
        eval_: bool = True,
        to_cpu: bool = False,
        num_workers: int = 0,
    ) -> Union[Tensor, np.ndarray]:
        output = self.predict(input, batch_size, False, eval_, False, to_cpu, num_workers)
        pmf = output2pmf(output)
        return tt.utils.array_or_tensor(pmf, numpy, input)

    def interpolate(
        self, sub: int = 10, scheme: str = "const_pdf", duration_index: Optional[np.ndarray] = None
    ) -> InterpolatePMF:
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

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: Optional[Union[str, int, torch.device]] = None,
        duration_index: Optional[np.ndarray] = None,
        loss: Optional[Callable] = None,
    ) -> None:
        if loss is None:
            loss = NLLPMFLoss()
        super().__init__(net, loss, optimizer, device, duration_index)


def output2surv(output: Tensor) -> Tensor:
    """Transform a network output tensor to discrete survival estimates.

    Ref: PMF
    """
    pmf = output2pmf(output)
    return pmf2surv(pmf)
