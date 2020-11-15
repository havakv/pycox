from typing import Any, Callable, Optional, Union

import numpy as np
import pandas as pd
import torch
import torchtuples as tt
from pycox import models
from pycox.models import discrete_time
from pycox.models.interpolation import InterpolateLogisticHazard
from pycox.preprocessing import label_transforms
from torch import Tensor


class LogisticHazard(models.base.SurvBase):
    """
    A discrete-time survival model that minimize the likelihood for right-censored data by
    parameterizing the hazard function. Also known as  "Nnet-survival" [3].

    The Logistic-Hazard was first proposed by [2], but this implementation follows [1].

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

    [2] Charles C. Brown. On the use of indicator variables for studying the time-dependence of parameters
        in a response-time model. Biometrics, 31(4):863–872, 1975.
        https://www.jstor.org/stable/2529811?seq=1#metadata_info_tab_contents

    [3] Michael F. Gensheimer and Balasubramanian Narasimhan. A scalable discrete-time survival model for
        neural networks. PeerJ, 7:e6257, 2019.
        https://peerj.com/articles/6257/
    """

    label_transform = label_transforms.LabTransDiscreteTime

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: Optional[Union[str, int, torch.device]] = None,
        duration_index: Optional[np.ndarray] = None,
        loss: Optional[Callable] = None,
    ) -> None:
        self.duration_index = duration_index
        if loss is None:
            loss = models.loss.NLLLogistiHazardLoss()
        super().__init__(net, loss, optimizer, device)

    @property
    def duration_index(self) -> np.ndarray:
        """
        Array of durations that defines the discrete times. This is used to set the index
        of the DataFrame in `predict_surv_df`.

        Returns:
            np.array -- Duration index.
        """
        return self._duration_index

    @duration_index.setter
    def duration_index(self, val: np.ndarray) -> None:
        self._duration_index = val

    def predict_surv_df(
        self,
        input: Any,
        batch_size: int = 8224,
        eval_: bool = True,
        num_workers: int = 0,
        epsilon: float = 1e-7,
    ) -> pd.DataFrame:
        surv = self.predict_surv(input, batch_size, True, eval_, True, num_workers, epsilon)
        return pd.DataFrame(surv.transpose(), self.duration_index)

    def predict_surv(
        self,
        input: Any,
        batch_size: int = 8224,
        numpy: Optional[bool] = None,
        eval_: bool = True,
        to_cpu: bool = False,
        num_workers: int = 0,
        epsilon: float = 1e-7,
    ) -> Union[Tensor, np.ndarray]:
        output = self.predict(input, batch_size, False, eval_, False, to_cpu, num_workers)
        surv = discrete_time.output2surv(output, epsilon)
        return tt.utils.array_or_tensor(surv, numpy, input)

    def predict_hazard(
        self,
        input: Any,
        batch_size: int = 8224,
        numpy: Optional[bool] = None,
        eval_: bool = True,
        to_cpu: bool = False,
        num_workers: int = 0,
    ) -> Union[Tensor, np.ndarray]:
        output = self.predict(input, batch_size, False, eval_, False, to_cpu, num_workers)
        hazard = discrete_time.output2hazard(output)
        return tt.utils.array_or_tensor(hazard, numpy, input)

    def interpolate(self, sub=10, scheme="const_pdf", duration_index=None):
        """Use interpolation for predictions.
        There are two schemes:
            `const_hazard` and `exp_surv` which assumes pice-wise constant hazard in each interval (exponential survival).
            `const_pdf` and `lin_surv` which assumes pice-wise constant PMF in each interval (linear survival).

        Keyword Arguments:
            sub {int} -- Number of "sub" units in interpolation grid. If `sub` is 10 we have a grid with
                10 times the number of grid points than the original `duration_index` (default: {10}).
            scheme {str} -- Type of interpolation {'const_hazard', 'const_pdf'}.
                See `InterpolateDiscrete` (default: {'const_pdf'})
            duration_index {np.array} -- Cuts used for discretization. Does not affect interpolation,
                only for setting index in `predict_surv_df` (default: {None})

        Returns:
            [InterpolateLogisticHazard] -- Object for prediction with interpolation.
        """
        if duration_index is None:
            duration_index = self.duration_index
        return InterpolateLogisticHazard(self, scheme, duration_index, sub)
