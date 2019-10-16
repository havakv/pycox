import pandas as pd
import numpy as np
import numba

def idx_at_times(index_surv, times, steps='pre', assert_sorted=True):
    """Gives index of `index_surv` corresponding to `time`, i.e. 
    `index_surv[idx_at_times(index_surv, times)]` give the values of `index_surv`
    closet to `times`.
    
    Arguments:
        index_surv {np.array} -- Durations of survival estimates
        times {np.array} -- Values one want to match to `index_surv`
    
    Keyword Arguments:
        steps {str} -- Round 'pre' (closest value higher) or 'post'
          (closest value lower) (default: {'pre'})
        assert_sorted {bool} -- Assert that index_surv is monotone (default: {True})
    
    Returns:
        np.array -- Index of `index_surv` that is closest to `times`
    """
    if assert_sorted:
        assert pd.Series(index_surv).is_monotonic_increasing, "Need 'index_surv' to be monotonic increasing"
    if steps == 'pre':
        idx = np.searchsorted(index_surv, times)
    elif steps == 'post':
        idx = np.searchsorted(index_surv, times, side='right') - 1
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
    survive = 1 - di / ni
    zero_survive = survive == 0
    if zero_survive.any():
        i = np.argmax(zero_survive)
        surv = np.zeros_like(survive)
        surv[:i] = np.exp(np.log(survive[:i]).cumsum())
        # surv[i:] = surv[i-1]
        surv[i:] = 0.
    else:
        surv = np.exp(np.log(1 - di / ni).cumsum())
    if start_duration != surv_idx.min():
        tmp = np.ones(len(surv)+ 1, dtype=surv.dtype)
        tmp[1:] = surv
        surv = tmp
        tmp = np.zeros(len(surv_idx)+ 1, dtype=surv_idx.dtype)
        tmp[1:] = surv_idx
        surv_idx = tmp
    surv = pd.Series(surv, surv_idx)
    return surv
