import numpy as np
import torch
import torchtuples as tt

def pad_col(phi, val=0):
    """Addes a column of `val` at the end of phi"""
    if len(phi.shape) != 2:
        raise ValueError(f"Only works for `phi` tensor that is 2-D.")
    phi_mp1 = torch.zeros_like(phi[:, :1])
    if val != 0:
        phi_mp1 = phi_mp1 + val
    return torch.cat([phi, phi_mp1], dim=1)

def array_or_tensor(tensor, numpy, input):
    """Returs a tensor if numpy is False or input is tensor.
    Else it returns numpy array.
    """
    if numpy is False:
        return tensor
    if (numpy is True) or (tt.tuplefy(input).type() is np.ndarray):
        tensor = tensor.cpu().numpy()
    return tensor
