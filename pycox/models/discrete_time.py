from torch import Tensor


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
