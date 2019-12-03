import warnings
import numpy as np
import torch
import torch.nn.functional as F
import torchtuples as tt

def pad_col(input, val=0, where='end'):
    """Addes a column of `val` at the start of end of `input`."""
    if len(input.shape) != 2:
        raise ValueError(f"Only works for `phi` tensor that is 2-D.")
    pad = torch.zeros_like(input[:, :1])
    if val != 0:
        pad = pad + val
    if where == 'end':
        return torch.cat([input, pad], dim=1)
    elif where == 'start':
        return torch.cat([pad, input], dim=1)
    raise ValueError(f"Need `where` to be 'start' or 'end', got {where}")

def array_or_tensor(tensor, numpy, input):
    warnings.warn('Use `torchtuples.utils.array_or_tensor` instead', DeprecationWarning)
    return tt.utils.array_or_tensor(tensor, numpy, input)

def make_subgrid(grid, sub=1):
    """When calling `predict_surv` with sub != 1 this can help with
    creating the duration index of the survival estimates.

    E.g.
    sub = 5
    surv = model.predict_surv(test_input, sub=sub)
    grid = model.make_subgrid(cuts, sub)
    surv = pd.DataFrame(surv, index=grid)
    """
    subgrid = tt.TupleTree(np.linspace(start, end, num=sub+1)[:-1]
                        for start, end in zip(grid[:-1], grid[1:]))
    subgrid = subgrid.apply(lambda x: tt.TupleTree(x)).flatten() + (grid[-1],)
    return subgrid

def log_softplus(input, threshold=-15.):
    """Equivalent to 'F.softplus(input).log()', but for 'input < threshold',
    we return 'input', as this is approximately the same.

    Arguments:
        input {torch.tensor} -- Input tensor
    
    Keyword Arguments:
        threshold {float} -- Treshold for when to just return input (default: {-15.})
    
    Returns:
        torch.tensor -- return log(softplus(input)).
    """
    output = input.clone()
    above = input >= threshold
    output[above] = F.softplus(input[above]).log()
    return output

def cumsum_reverse(input: torch.Tensor, dim: int = 1) -> torch.Tensor:
    if dim != 1:
        raise NotImplementedError
    input = input.sum(1, keepdim=True) - pad_col(input, where='start').cumsum(1)
    return input[:, :-1]
