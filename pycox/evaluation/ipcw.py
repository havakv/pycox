import numpy as np
import scipy
import numba
from pycox import utils

@numba.njit(parallel=True)
def _inv_cens_scores(func, time_grid, durations, events, surv, censor_surv, idx_ts_surv, idx_ts_censor,
                     idx_tt_censor, scores, weights, n_times, n_indiv, max_weight):
    def _inv_cens_score_single(func, ts, durations, events, surv, censor_surv, idx_ts_surv_i,
                               idx_ts_censor_i, idx_tt_censor, scores, weights, n_indiv, max_weight):
        min_g = 1./max_weight
        for i in range(n_indiv):
            tt = durations[i]
            d = events[i]
            s = surv[idx_ts_surv_i, i]
            g_ts = censor_surv[idx_ts_censor_i, i]
            g_tt = censor_surv[idx_tt_censor[i], i]
            g_ts = max(g_ts, min_g)
            g_tt = max(g_tt, min_g)
            score, w = func(ts, tt, s, g_ts, g_tt, d)
            #w = min(w, max_weight)
            scores[i] = score * w
            weights[i] = w

    for i in numba.prange(n_times):
        ts = time_grid[i]
        idx_ts_surv_i = idx_ts_surv[i]
        idx_ts_censor_i = idx_ts_censor[i]
        scores_i = scores[i]
        weights_i = weights[i]
        _inv_cens_score_single(func, ts, durations, events, surv, censor_surv, idx_ts_surv_i,
                               idx_ts_censor_i, idx_tt_censor, scores_i, weights_i, n_indiv, max_weight)

def _inverse_censoring_weighted_metric(func):
    if not func.__class__.__module__.startswith('numba'):
        raise ValueError("Need to provide numba compiled function")
    def metric(time_grid, durations, events, surv, censor_surv, index_surv, index_censor, max_weight=np.inf,
               reduce=True, steps_surv='post', steps_censor='post'):
        if not hasattr(time_grid, '__iter__'):
            time_grid = np.array([time_grid])
        assert (type(time_grid) is type(durations) is type(events) is type(surv) is type(censor_surv) is
                type(index_surv) is type(index_censor) is np.ndarray), 'Need all input to be np.ndarrays'
        n_times = len(time_grid)
        n_indiv = len(durations)
        scores = np.zeros((n_times, n_indiv))
        weights = np.zeros((n_times, n_indiv))
        idx_ts_surv = utils.idx_at_times(index_surv, time_grid, steps_surv, assert_sorted=True)
        idx_ts_censor = utils.idx_at_times(index_censor, time_grid, steps_censor, assert_sorted=True)
        idx_tt_censor = utils.idx_at_times(index_censor, durations, 'pre', assert_sorted=True)
        if steps_censor == 'post':
            idx_tt_censor  = (idx_tt_censor - 1).clip(0)
            #  This ensures that we get G(tt-)
        _inv_cens_scores(func, time_grid, durations, events, surv, censor_surv, idx_ts_surv, idx_ts_censor,
                         idx_tt_censor, scores, weights, n_times, n_indiv, max_weight)
        if reduce is True:
            return np.sum(scores, axis=1) / np.sum(weights, axis=1)
        return scores, weights
    return metric

@numba.njit()
def _brier_score(ts, tt, s, g_ts, g_tt, d):
    if (tt <= ts) and d == 1:
        return np.power(s, 2), 1./g_tt
    if tt > ts:
        return np.power(1 - s, 2), 1./g_ts
    return 0., 0.

@numba.njit()
def _binomial_log_likelihood(ts, tt, s, g_ts, g_tt, d, eps=1e-7):
    s = eps if s < eps else s
    s = (1-eps) if s > (1 - eps) else s
    if (tt <= ts) and d == 1:
        return np.log(1 - s), 1./g_tt
    if tt > ts:
        return np.log(s), 1./g_ts
    return 0., 0.

brier_score = _inverse_censoring_weighted_metric(_brier_score)
binomial_log_likelihood = _inverse_censoring_weighted_metric(_binomial_log_likelihood)

def _integrated_inverce_censoring_weighed_metric(func):
    def metric(time_grid, durations, events, surv, censor_surv, index_surv, index_censor,
               max_weight=np.inf, steps_surv='post', steps_censor='post'):
        scores = func(time_grid, durations, events, surv, censor_surv, index_surv, index_censor,
                      max_weight, True, steps_surv, steps_censor)
        integral = scipy.integrate.simpson(scores, time_grid)
        return integral / (time_grid[-1] - time_grid[0])
    return metric

integrated_brier_score = _integrated_inverce_censoring_weighed_metric(brier_score)
integrated_binomial_log_likelihood = _integrated_inverce_censoring_weighed_metric(binomial_log_likelihood)
