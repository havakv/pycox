from torch import Tensor

from pycox.models import utils


def hazard2surv(hazard: Tensor, epsilon: float = 1e-7) -> Tensor:
    """Transform discrete hazards to discrete survival estimates.

    Ref: LogisticHazard
    """
    return (1 - hazard).add(epsilon).log().cumsum(1).exp()


def output2hazard(output: Tensor) -> Tensor:
    """Transform a network output tensor to discrete hazards. This just calls the sigmoid function

    Ref: LogisticHazard
    """
    return output.sigmoid()


def pmf2surv(pmf: Tensor) -> Tensor:
    """Transform discrete PMF to discrete survival estimates.

    Ref: PMF
    """
    return 1.0 - pmf.cumsum(1)


def output2pmf(output: Tensor) -> Tensor:
    """Transform a network output tensor to discrete PMF.

    Ref: PMF
    """
    return utils.pad_col(output).softmax(1)[:, :-1]
