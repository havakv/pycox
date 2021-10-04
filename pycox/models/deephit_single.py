from typing import Any, Callable, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import torchtuples as tt
from torchtuples.data import DataLoaderBatch

from pycox import models
from pycox.models.data import pair_rank_mat, DeepHitDataset
from pycox.models.loss import DeepHitSingleLoss
from pycox.models.pmf import output2pmf, output2surv, pmf2surv
from pycox.preprocessing.label_transforms import LabTransDiscreteTime


class DeepHitSingle(models.pmf.PMFBase):
    """The DeepHit methods by [1] but only for single event (not competing risks).

    Note that `alpha` is here defined differently than in [1], as `alpha` is  weighting between
    the likelihood and rank loss (see Appendix D in [2])
        loss = alpha * nll + (1 - alpha) rank_loss(sigma).

    Also, unlike [1], this implementation allows for survival past the max durations, i.e., it
    does not assume all events happen within the defined duration grid. See [3] for details.

    Keyword Arguments:
        alpha {float} -- Weighting (0, 1) likelihood and rank loss (L2 in paper).
            1 gives only likelihood, and 0 gives only rank loss. (default: {0.2})
        sigma {float} -- from eta in rank loss (L2 in paper) (default: {0.1})

    References:
    [1] Changhee Lee, William R Zame, Jinsung Yoon, and Mihaela van der Schaar. Deephit: A deep learning
        approach to survival analysis with competing risks. In Thirty-Second AAAI Conference on Artificial
        Intelligence, 2018.
        http://medianetlab.ee.ucla.edu/papers/AAAI_2018_DeepHit

    [2] Håvard Kvamme, Ørnulf Borgan, and Ida Scheel.
        Time-to-event prediction with neural networks and Cox regression.
        Journal of Machine Learning Research, 20(129):1–30, 2019.
        http://jmlr.org/papers/v20/18-424.html

    [3] Håvard Kvamme and Ørnulf Borgan. Continuous and Discrete-Time Survival Prediction
        with Neural Networks. arXiv preprint arXiv:1910.06724, 2019.
        https://arxiv.org/pdf/1910.06724.pdf
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: Optional[Union[str, int, torch.device]] = None,
        duration_index: Optional[np.ndarray] = None,
        alpha: float = 0.2,
        sigma: float = 0.1,
        loss: Optional[Callable] = None,
    ) -> None:
        if loss is None:
            loss = DeepHitSingleLoss(alpha, sigma)
        super().__init__(net, loss, optimizer, device, duration_index)

    @staticmethod
    def make_dataloader(data: Any, batch_size: int, shuffle: bool, num_workers: int = 0) -> DataLoaderBatch:
        return tt.make_dataloader(data, batch_size, shuffle, num_workers, make_dataset=DeepHitDataset)

    @staticmethod
    def make_dataloader_predict(
        input: Any, batch_size: int, shuffle: bool = False, num_workers: int = 0
    ) -> DataLoaderBatch:
        return tt.make_dataloader(input, batch_size, shuffle, num_workers)
