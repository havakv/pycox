from typing import Optional, Union, Tuple, Any, Callable, List
import numpy as np
import pandas as pd
import torch
from torch import Tensor
import pycox.models.base as base
from pycox.models.loss import CoxPHLoss
import torchtuples as tt


class CoxPH(base.SurvBase):
    """Cox proportional hazards model parameterized with a neural net.
    This is essentially the DeepSurv method [1].

    The loss function is not quite the partial log-likelihood, but close.
    The difference is that for tied events, we use a random order instead of
    including all individuals that had an event at that point in time.

    Arguments:
        net {torch.nn.Module} -- A pytorch net.

    Keyword Arguments:
        optimizer {torch or torchtuples optimizer} -- Optimizer (default: {None})
        device {str, int, torch.device} -- Device to compute on. (default: {None})
            Preferably pass a torch.device object.
            If 'None': use default gpu if available, else use cpu.
            If 'int': used that gpu: torch.device('cuda:<device>').
            If 'string': string is passed to torch.device('string').

    [1] Jared L. Katzman, Uri Shaham, Alexander Cloninger, Jonathan Bates, Tingting Jiang, and Yuval Kluger.
        Deepsurv: personalized treatment recommender system using a Cox proportional hazards deep neural network.
        BMC Medical Research Methodology, 18(1), 2018.
        https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/s12874-018-0482-1
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: Any = None,
        device: Optional[Union[int, str]] = None,
        loss: Callable = None,
    ):
        if loss is None:
            loss = CoxPHLoss()
        super().__init__(net, loss, optimizer, device)

    @property
    def duration_index(self) -> np.ndarray:
        return self._duration_index.numpy()

    @property
    def baseline_hazards(self) -> np.ndarray:
        return self._baseline_hazards.numpy()

    @property
    def cumulative_baseline_hazards(self) -> np.ndarray:
        return self._cumulative_baseline_hazards.numpy()

    def fit(
        self,
        input: Any,
        target: Any,
        batch_size: int = 256,
        epochs: int = 1,
        callbacks: Optional[List[tt.cb.Callback]] = None,
        verbose: bool = True,
        num_workers: int = 0,
        shuffle: bool = True,
        metrics: Optional[Any] = None,
        val_data: Optional[Any] = None,
        val_batch_size: int = 8224,
        **kwargs,
    ) -> tt.cb.TrainingLogger:
        self.train_data = tt.tuplefy(input, target)
        return super().fit(
            input,
            target,
            batch_size,
            epochs,
            callbacks,
            verbose,
            num_workers,
            shuffle,
            metrics,
            val_data,
            val_batch_size,
            **kwargs,
        )

    def compute_baseline_hazards(
        self,
        input: Optional[Any] = None,
        target: Optional[Any] = None,
        batch_size: int = 8224,
        numpy: Optional[bool] = None,
        eval_: bool = True,
        num_workers: int = 0,
        set_hazards: bool = True,
        sample: Optional[Union[int, float]] = None,
    ) -> Tuple[Union[np.ndarray, Tensor]]:
        if input is None and target is None:
            if self.train_data is None:
                raise ValueError("Need to give a 'input' and 'target' to this function.")
            input, target = self.train_data
        if sample is not None:
            if sample < 1:
                num_samples = len(target[0])
                sample = int(num_samples * sample)
            idx = np.random.choice(num_samples, sample, replace=False)
            input, target = tt.tuplefy(input, target).iloc[idx]
        output = self.predict(input, batch_size, False, eval_, False, True, num_workers)
        return self._compute_baseline_hazards(input, target, output, set_hazards, numpy)

    def _compute_baseline_hazards(
        self, input: Any, target: Any, output: Tensor, set_hazards: bool, numpy
    ) -> Tuple[Union[np.ndarray, Tensor]]:
        target = tt.tuplefy(target).to_tensor()
        baseline_hazards, durations = compute_baseline_hazards(output, *target)
        if set_hazards:
            self._duration_index = durations
            self._baseline_hazards = baseline_hazards
            self._cumulative_baseline_hazards = baseline_hazards.cumsum(0)
        return tt.utils.array_or_tensor(baseline_hazards, numpy, input)

    def compute_baseline_hazards_dataloader(
        self,
        dataloader: Optional[Any] = None,
        numpy: Optional[bool] = None,
        eval_: bool = True,
        set_hazards: bool = True,
    ) -> Tuple[Union[np.ndarray, Tensor]]:
        # Need to get output (easy), and targe (not that easy)
        raise NotImplementedError

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
        if self._cumulative_baseline_hazards is None:
            raise ValueError("Need to compute baseline hazards to predict")
        output = self.predict(input, batch_size, False, eval_, False, to_cpu, num_workers)
        surv = output2surv(output, self._cumulative_baseline_hazards)
        return tt.utils.array_or_tensor(surv, numpy, input)


def compute_baseline_hazards(
    output: Tensor, durations: Tensor, events: Tensor
) -> Tuple[Tensor, Tensor]:
    durations, idx = torch.sort(durations.flatten(), descending=False)
    events = events.flatten()[idx]
    output = output.flatten()[idx]

    # Get mask of last index before the duration changes to a new value
    changed = torch.ones_like(durations, dtype=torch.bool)
    changed[:-1] = durations[1:] - durations[:-1]

    cum_events = events.cumsum(0)[changed]
    event_counts = torch.zeros_like(cum_events, dtype=torch.int64)
    event_counts[1:] = cum_events[1:] - cum_events[:-1]
    event_counts[0] = cum_events[0]
    # evnt_counts is the number of events for each unique value of duration

    rev_cum_exp_out = output.exp().flip(0).cumsum(0).flip(0)[changed]
    baseline_hazards = event_counts / rev_cum_exp_out
    return baseline_hazards, durations[changed]


def compute_cumulative_baseline_hazards(
    output: Tensor, durations: Tensor, events: Tensor
) -> Tuple[Tensor, Tensor]:
    baseline_hazards, unique_durations = compute_baseline_hazards(output, durations, events)
    return baseline_hazards.cumsum(0), unique_durations


def output2cumulative_hazards(output: Tensor, cumulative_baseline_hazards: Tensor) -> Tensor:
    return output.exp().reshape(-1, 1).mm(cumulative_baseline_hazards.reshape(1, -1))


def output2surv(output: Tensor, cumulative_baseline_hazards: Tensor) -> Tensor:
    cumulative_hazards = output2cumulative_hazards(output, cumulative_baseline_hazards)
    return torch.exp(-cumulative_hazards)
