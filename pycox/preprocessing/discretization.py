import warnings
import numpy as np
import pandas as pd
from pycox import utils


def make_cuts(n_cuts, scheme, durations, events, min_=0., dtype='float64'):
    if scheme == 'equidistant':
        cuts = cuts_equidistant(durations.max(), n_cuts, min_, dtype)
    elif scheme == 'quantiles':
        cuts = cuts_quantiles(durations, events, n_cuts, min_, dtype)
    else:
        raise ValueError(f"Got invalid `scheme` {scheme}.")
    if (np.diff(cuts) == 0).any():
        raise ValueError("cuts are not unique.")
    return cuts

def _values_if_series(x):
    if type(x) is pd.Series:
        return x.values
    return x

def cuts_equidistant(max_, num, min_=0., dtype='float64'):
    return np.linspace(min_, max_, num, dtype=dtype)

def cuts_quantiles(durations, events, num, min_=0., dtype='float64'):
    """
    If min_ = None, we will use durations.min() for the first cut.
    """
    km = utils.kaplan_meier(durations, events)
    surv_est, surv_durations = km.values, km.index.values
    s_cuts = np.linspace(km.values.min(), km.values.max(), num)
    cuts_idx = np.searchsorted(surv_est[::-1], s_cuts)[::-1]
    cuts = surv_durations[::-1][cuts_idx]
    cuts = np.unique(cuts)
    if len(cuts) != num:
        warnings.warn(f"cuts are not unique, continue with {len(cuts)} cuts instead of {num}")
    cuts[0] = durations.min() if min_ is None else min_
    assert cuts[-1] == durations.max(), 'something wrong...'
    return cuts.astype(dtype)

def _is_monotonic_increasing(x):
    assert len(x.shape) == 1, 'Only works for 1d'
    return (x[1:] >= x[:-1]).all()

def bin_numerical(x, right_cuts, error_on_larger=False):
    """
    Discretize x into bins defined by right_cuts (needs to be sorted).
    If right_cuts = [1, 2], we have bins (-inf, 1], (1, 2], (2, inf).
    error_on_larger results in a ValueError if x contains larger
    values than right_cuts.
    
    Returns index of bins.
    To optaine values do righ_cuts[bin_numerica(x, right_cuts)].
    """
    assert _is_monotonic_increasing(right_cuts), 'Need `right_cuts` to be sorted.'
    bins = np.searchsorted(right_cuts, x, side='left')
    if bins.max() == right_cuts.size:
        if error_on_larger:
            raise ValueError('x contrains larger values than right_cuts.')
    return bins

def discretize(x, cuts, side='right', error_on_larger=False):
    """Discretize x to cuts.
    
    Arguments:
        x {np.array} -- Array of times.
        cuts {np.array} -- Sortet array of discrete times.
    
    Keyword Arguments:
        side {str} -- If we shold round down or up (left, right) (default: {'right'})
        error_on_larger {bool} -- If we shold return an error if we pass higher values
            than cuts (default: {False})
    
    Returns:
        np.array -- Discretized values.
    """
    if side not in ['right', 'left']:
        raise ValueError('side argument needs to be right or left.')
    bins = bin_numerical(x, cuts, error_on_larger)
    if side == 'right':
        cuts = np.concatenate((cuts, np.array([np.inf])))
        return cuts[bins]
    bins_cut = bins.copy()
    bins_cut[bins_cut == cuts.size] = -1
    exact = cuts[bins_cut] == x
    left_bins = bins - 1 + exact
    vals = cuts[left_bins]
    vals[left_bins == -1] = - np.inf
    return vals


class _OnlyTransform:
    """Abstract class for sklearn preprocessing methods.
    Only implements fit and fit_transform.
    """
    def fit(self, *args):
        return self
    
    def transform(self, *args):
        raise NotImplementedError
    
    def fit_transform(self, *args):
        return self.fit(*args).transform(*args)


class DiscretizeUnknownC(_OnlyTransform):
    """Implementation of scheme 2.
    
    cuts should be [t0, t1, ..., t_m], where t_m is right sensored value.
    """
    def __init__(self, cuts, right_censor=False, censor_side='left'):
        self.cuts = cuts
        self.right_censor = right_censor
        self.censor_side = censor_side
    
    def transform(self, duration, event):
        dtype_event = event.dtype
        event = event.astype('bool')
        if self.right_censor:
            duration = duration.copy()
            censor = duration > self.cuts.max()
            duration[censor] = self.cuts.max()
            event[censor] = False
        if duration.max() > self.cuts.max():
            raise ValueError("`duration` contains larger values than cuts. Set `right_censor`=True to censor these")
        td = np.zeros_like(duration)
        c = event == False
        td[event] = discretize(duration[event], self.cuts, side='right', error_on_larger=True)
        if c.any():
            td[c] = discretize(duration[c], self.cuts, side=self.censor_side, error_on_larger=True)
        return td, event.astype(dtype_event)


def duration_idx_map(duration):
    duration = np.unique(duration)
    duration = np.sort(duration)
    idx = np.arange(duration.shape[0])
    return {d: i for i, d in zip(idx, duration)}


class Duration2Idx(_OnlyTransform):
    def __init__(self, durations=None):
        self.durations = durations
        if durations is None:
            raise NotImplementedError()
        if self.durations is not None:
            self.duration_to_idx = self._make_map(self.durations)
    
    @staticmethod
    def _make_map(durations):
        return np.vectorize(duration_idx_map(durations).get)
    
    def transform(self, duration, y=None):
        if duration.dtype is not self.durations.dtype:
            raise ValueError('Need `time` to have same type as `self.durations`.')
        idx = self.duration_to_idx(duration)
        if np.isnan(idx).any():
            raise ValueError('Encountered `nans` in transformed indexes.')
        return idx


class IdxDiscUnknownC:
    """Get indexed for discrete data using cuts.

        Arguments:
            cuts {np.array} -- Array or right cuts.
        
        Keyword Arguments:
            label_cols {tuple} -- Name of label columns in dataframe (default: {None}).
    """
    def __init__(self, cuts, label_cols=None, censor_side='left'):
        self.cuts = cuts
        self.duc = DiscretizeUnknownC(cuts, right_censor=True, censor_side=censor_side)
        self.di = Duration2Idx(cuts)
        self.label_cols= label_cols
    
    def transform(self, time, d):
        time, d = self.duc.transform(time, d)
        idx = self.di.transform(time)
        return idx, d
    
    def transform_df(self, df):
        if self.label_cols is None:
            raise RuntimeError("Need to set 'label_cols' to use this. Use 'transform instead'")
        col_duration, col_event = self.label_cols
        time = df[col_duration].values
        d = df[col_event].values
        return self.transform(time, d)
