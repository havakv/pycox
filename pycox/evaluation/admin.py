import numpy as np
import scipy
import numba
from pycox.utils import idx_at_times


def administrative_scores(func):
    if not func.__class__.__module__.startswith('numba'):
        raise ValueError("Need to provide numba compiled function")
    def metric(time_grid, durations, durations_c, events, surv, index_surv, reduce=True, steps_surv='post'):
        if not hasattr(time_grid, '__iter__'):
            time_grid = np.array([time_grid])
        assert (type(time_grid) is type(durations) is type(events) is type(surv) is
                type(index_surv) is type(durations_c) is np.ndarray), 'Need all input to be np.ndarrays'
        assert (durations[events == 0] == durations_c[events == 0]).all(), 'Censored observations need same `durations` and `durations_c`'
        assert (durations[events == 1] <= durations_c[events == 1]).all(), '`durations` cannot be larger than `durations_c`'
        idx_ts_surv = idx_at_times(index_surv, time_grid, steps_surv, assert_sorted=True)
        scores, norm = _admin_scores(func, time_grid, durations, durations_c, events, surv, idx_ts_surv)
        if reduce is True:
            return scores.sum(axis=1) / norm
        return scores, norm.reshape(-1, 1)
    return metric

@numba.njit(parallel=True)
def _admin_scores(func, time_grid, durations, durations_c, events, surv, idx_ts_surv):
    def _single(func, ts, durations, durations_c, events, surv, idx_ts_surv_i,
                scores, n_indiv):
        for i in range(n_indiv):
            tt = durations[i]
            tc = durations_c[i]
            d = events[i]
            s = surv[idx_ts_surv_i, i]
            scores[i] = func(ts, tt, tc, d, s)

    n_times = len(time_grid)
    n_indiv = len(durations)
    scores = np.empty((n_times, n_indiv))
    scores.fill(np.nan)
    normalizer = np.empty(n_times)
    normalizer.fill(np.nan)
    for i in numba.prange(n_times):
        ts = time_grid[i]
        idx_ts_surv_i = idx_ts_surv[i]
        scores_i = scores[i]
        normalizer[i] = (durations_c >= ts).sum()
        _single(func, ts, durations, durations_c, events, surv, idx_ts_surv_i, scores_i, n_indiv)
    return scores, normalizer

@numba.njit
def _brier_score(ts, tt, tc, d, s):
    if (tt <= ts) and (d == 1) and (tc >= ts):
        return np.power(s, 2)
    if tt >= ts:
        return np.power(1 - s, 2)
    return 0.

@numba.njit
def _binomial_log_likelihood(ts, tt, tc, d, s, eps=1e-7):
    if s < eps:
        s = eps
    elif s > (1 - eps):
        s = 1 - eps
    if (tt <= ts) and (d == 1) and (tc >= ts):
        return np.log(1 - s)
    if tt >= ts:
        return np.log(s)
    return 0.

brier_score = administrative_scores(_brier_score)
binomial_log_likelihood = administrative_scores(_binomial_log_likelihood)


def _integrated_admin_metric(func):
    def metric(time_grid, durations, durations_c, events, surv, index_surv, steps_surv='post'):
        scores = func(time_grid, durations, durations_c, events, surv, index_surv, True, steps_surv)
        integral = scipy.integrate.simps(scores, time_grid)
        return integral / (time_grid[-1] - time_grid[0])
    return metric

integrated_brier_score = _integrated_admin_metric(brier_score)
integrated_binomial_log_likelihood = _integrated_admin_metric(binomial_log_likelihood)

