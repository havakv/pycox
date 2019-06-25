import pandas as pd
import numpy as np
import numba

def idx_at_times(index_surv, times, assert_sorted=True):
    if assert_sorted:
        assert pd.Series(index_surv).is_monotonic_increasing, "Need 'index_surv' to be monotonic increasing"
    idx = np.searchsorted(index_surv, times)
    return idx.clip(0, len(index_surv)-1)

@numba.njit
def _group_loop(n, surv_idx, durations, events, di, ni):
    idx = 0
    for i in range(n):
        idx += durations[i] != surv_idx[idx]
        di[idx] += events[i]
        ni[idx] += 1
    return di, ni

def kaplan_meier(durations, events, start_duration=0):
    """A very simple Kaplan-Meier fitter. For a more complete implementation
    see `lifelines`.
    
    Arguments:
        durations {np.array} -- durations array
        events {np.arrray} -- events array 0/1
    
    Keyword Arguments:
        start_duration {int} -- Time start as `start_duration`. (default: {0})
    
    Returns:
        pd.Series -- Kaplan-Meier estimates.
    """
    n = len(durations)
    assert n == len(events)
    order = np.argsort(durations)
    durations = durations[order]
    events = events[order]
    surv_idx = np.unique(durations)
    ni = np.zeros(len(surv_idx), dtype='int')
    di = np.zeros_like(ni)
    di, ni = _group_loop(n, surv_idx, durations, events, di, ni)
    ni = n - ni.cumsum()
    ni[1:] = ni[:-1]
    ni[0] = n
    haz = 1 - di / ni
    zero_haz = haz == 0
    if zero_haz.any():
        i = np.argmax(zero_haz)
        surv = np.zeros_like(haz)
        surv[:i] = np.exp(np.log(haz[:i]).cumsum())
        surv[i:] = surv[i-1]
    else:
        surv = np.exp(np.log(1 - di / ni).cumsum())
    # surv = np.exp(np.log(1 - di / ni).cumsum())
    if start_duration != surv_idx.min():
        tmp = np.ones(len(surv)+ 1, dtype=surv.dtype)
        tmp[1:] = surv
        surv = tmp
        tmp = np.zeros(len(surv_idx)+ 1, dtype=surv_idx.dtype)
        tmp[1:] = surv_idx
        surv_idx = tmp
    surv = pd.Series(surv, surv_idx)
    return surv