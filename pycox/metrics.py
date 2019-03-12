'''
Some relevant metrics
'''
import warnings

import numpy as np
import scipy
import pandas as pd
import numba

from lifelines.utils import concordance_index
from lifelines import KaplanMeierFitter

def brier_score(times, prob_alive, durations, events):
    '''Compute the brier scores (for survival) at given times.

    For a specification on brier scores for survival data see e.g.:
    "Assessment of evaluation criteria for survival prediction from
    genomic data" by Bovelstad and Borgan.

    Parameters:
        times: Number or iterable with times where to compute the brier scores.
        prob_alive: Numpy array [len(times), len(durations)] with the estimated probabilities
            of each individual to be alive at each time in `times`. Each row represets
            a time in input array `times`.
        durations: Numpy array with time of events.
        events: Boolean numpy array indecating if dead/censored (True/False).

    Returns:
        Numpy array with brier scores.
    '''
    if not hasattr(times, '__iter__'):
        times = [times]
    assert prob_alive.__class__ is np.ndarray, 'Need numpy array'
    assert prob_alive.shape == (len(times), len(durations)),\
        'Need prob_alive to have dims [len(times), len(durations)].'
    kmf_censor = KaplanMeierFitter()
    kmf_censor.fit(durations, 1-events)
    # km_censor_at_durations = kmf_censor.predict(durations)
    km_censor_at_durations = kmf_censor.survival_function_.loc[durations].values.flatten()
    km_censor_at_times = kmf_censor.predict(times)

    def compute_score(time_, km_censor_at_time, prob_alive_):
        died = ((durations <= time_) & (events == True))
        survived = (durations > time_)
        event_part = (prob_alive_**2)[died] / km_censor_at_durations[died]
        survived_part = ((1 - prob_alive_)**2)[survived] / km_censor_at_time
        return (np.sum(event_part) + np.sum(survived_part)) / len(durations)

    b_scores = [compute_score(time_, km, pa)
                for time_, km, pa in zip(times, km_censor_at_times, prob_alive)]
    return np.array(b_scores)

def integrated_brier_score_numpy(times_grid, prob_alive, durations, events):
    '''Compute the integrated brier score (for survival).
    This funcion takes pre-computed probabilities, while the function integrated_brier_score
    takes a function and a grid instead.

    For a specification on brier scores for survival data see e.g.:
    "Assessment of evaluation criteria for survival prediction from
    genomic data" by Bovelstad and Borgan.

    Parameters:
        times_grid: Iterable with times where to compute the brier scores.
            Needs to be strictly increasing.
        prob_alive: Numpy array [len(times_grid), len(durations)] with the estimated
            probabilities of each individual to be alive at each time in `times_grid`.
            Each row represets a time in input array `times_grid`.
        durations: Numpy array with time of events.
        events: Boolean numpy array indecating if dead/censored (True/False).
    '''
    assert pd.Series(times_grid).is_monotonic_increasing,\
        'Need monotonic increasing times_grid.'
    b_scores = brier_score(times_grid, prob_alive, durations, events)
    is_finite = np.isfinite(b_scores)
    b_scores = b_scores[is_finite]
    times_grid = times_grid[is_finite]
    integral = scipy.integrate.simps(b_scores, times_grid)
    return integral / (times_grid[-1] - times_grid[0])

def integrated_brier_score(prob_alive_func, durations, events,
                           times_grid=None, n_grid_points=100):
    '''Compute the integrated brier score (for survival).
    This takes a function and a grid, while the function integrated_brier_score_numpy
    takes pre-computed probabilities instead.

    For a specification on brier scores for survival data see e.g.:
    "Assessment of evaluation criteria for survival prediction from
    genomic data" by Bovelstad and Borgan.

    Parameters:
        prob_alive_func: Function that takes an array of times and returns
            a matrix [len(times_grid), len(durations)] with survival probabilities.
        durations: Numpy array with time of events.
        events: Boolean numpy array indecating if dead/censored (True/False).
        times_grid: Specified time grid for integration. If None: use equidistant between
            smallest and largest value times of durations.
        n_grid_points: Only apply if grid is None. Gives number of grid poinst used
            in equidistant grid.
    '''
    if times_grid is None:
        times_grid = np.linspace(durations.min(), durations.max(), n_grid_points)
    prob_alive = prob_alive_func(times_grid)
    return integrated_brier_score_numpy(times_grid, prob_alive, durations, events)

def binomial_log_likelihood(times, prob_alive, durations, events, eps=1e-7):
    '''Compute the binomial log-likelihood for survival at given times.

    We compute binomial log-likelihood weighted by the inverse censoring distribution.
    This is the same weighting scheeme as for the brier score.

    Parameters:
        times: Number or iterable with times where to compute the brier scores.
        prob_alive: Numpy array [len(times), len(durations)] with the estimated probabilities
            of each individual to be alive at each time in `times`. Each row represets
            a time in input array `times`.
        durations: Numpy array with time of events.
        events: Boolean numpy array indecating if dead/censored (True/False).
        eps: Clip prob_alive at (eps, 1-eps).

    Returns:
        Numpy array with brier scores.
    '''
    if not hasattr(times, '__iter__'):
        times = [times]
    assert prob_alive.__class__ is np.ndarray, 'Need numpy array'
    assert prob_alive.shape == (len(times), len(durations)),\
        'Need prob_alive to have dims [len(times), len(durations)].'
    kmf_censor = KaplanMeierFitter()
    kmf_censor.fit(durations, 1-events)
    km_censor_at_durations = kmf_censor.survival_function_.loc[durations].values.flatten()
    km_censor_at_times = kmf_censor.predict(times)

    prob_alive = np.clip(prob_alive, eps, 1-eps)

    def compute_score(time_, km_censor_at_time, prob_alive_):
        died = ((durations <= time_) & (events == True))
        survived = (durations > time_)
        event_part = np.log(1 - prob_alive_[died]) / km_censor_at_durations[died]
        survived_part = np.log(prob_alive_[survived]) / km_censor_at_time
        return (np.sum(event_part) + np.sum(survived_part)) / len(durations)

    scores = [compute_score(time_, km, pa)
              for time_, km, pa in zip(times, km_censor_at_times, prob_alive)]
    return np.array(scores)

def integrated_binomial_log_likelihood_numpy(times_grid, prob_alive, durations, events):
    '''Compute the integrated brier score (for survival).
    This funcion takes pre-computed probabilities, while the function integrated_brier_score
    takes a function and a grid instead.

    For a specification on brier scores for survival data see e.g.:
    "Assessment of evaluation criteria for survival prediction from
    genomic data" by Bovelstad and Borgan.

    Parameters:
        times_grid: Iterable with times where to compute the brier scores.
            Needs to be strictly increasing.
        prob_alive: Numpy array [len(times_grid), len(durations)] with the estimated
            probabilities of each individual to be alive at each time in `times_grid`.
            Each row represets a time in input array `times_grid`.
        durations: Numpy array with time of events.
        events: Boolean numpy array indecating if dead/censored (True/False).
    '''
    assert pd.Series(times_grid).is_monotonic_increasing,\
        'Need monotonic increasing times_grid.'
    scores = binomial_log_likelihood(times_grid, prob_alive, durations, events)
    is_finite = np.isfinite(scores)
    scores = scores[is_finite]
    times_grid = times_grid[is_finite]
    integral = scipy.integrate.simps(scores, times_grid)
    return integral / (times_grid[-1] - times_grid[0])

@numba.jit(nopython=True)
def _is_comparable(t_i, t_j, d_i, d_j):
    return ((t_i < t_j) & d_i) | ((t_i == t_j) & d_i & (d_j == 0))

@numba.jit(nopython=True)
def _is_concordant(s_i, s_j, t_i, t_j, d_i, d_j):
    return (s_i < s_j) & _is_comparable(t_i, t_j, d_i, d_j)

@numba.jit(nopython=True, parallel=True)
def _sum_comparable(t, d):
    n = t.shape[0]
    count = 0
    for i in numba.prange(n):
        for j in range(n):
            count += _is_comparable(t[i], t[j], d[i], d[j])
    return count

@numba.jit(nopython=True, parallel=True)
def _sum_concordant(s, t, d):
    n = len(t)
    count = 0
    for i in numba.prange(n):
        for j in range(n):
            count += _is_concordant(s[i, i], s[i, j], t[i], t[j], d[i], d[j])
    return count

def concordance_td(event_time, event, prob_alive):
    """Time dependent concorance index from 
    Antolini, L.; Boracchi, P.; and Biganzoli, E. 2005. A timedependent discrimination
    index for survival data. Statistics in Medicine 24:3927–3944.

    Arguments:
        event_time {np.array[n]} -- Event times (or censoring times.)
        event {np.array[n]} -- Event indicators (0 is censoring).
        prob_alive {np.array[n, n]} -- Survival probabilities n x n matrix, s.t.
            prob_alive[i, j] gives survial prob at event_time[i] for individual j.

    Returns:
        float -- Time dependent concordance index.
    """
    assert prob_alive.shape[0] == prob_alive.shape[1] == event_time.shape[0] == event.shape[0]
    assert type(event_time) is type(event) is type(prob_alive) is np.ndarray
    if event.dtype in ('float', 'float32'):
        event = event.astype('int32')
    return _sum_concordant(prob_alive, event_time, event) / _sum_comparable(event_time, event)

@numba.jit(nopython=True, parallel=True)
def _sum_concordant_disc(s, t, d, s_idx):
    n = len(t)
    count = 0
    for i in numba.prange(n):
        idx = s_idx[i]
        for j in range(n):
            count += _is_concordant(s[idx, i], s[idx, j], t[i], t[j], d[i], d[j])
    return count

def concordance_td_disc(event_time, event, surv_func, surv_idx):
    """Smaller memory (possibly) time dependent concorance index from 
    Antolini, L.; Boracchi, P.; and Biganzoli, E. 2005. A timedependent discrimination
    index for survival data. Statistics in Medicine 24:3927–3944.

    This method works well when the number of distinct event times in the training set
    is much smaller than in the test set. Instead of calculating all prob_alive
    (as in concordance_td), we give and surv_idx used to get prob_alive form surv_func.

    Arguments:
        event_time {np.array[n]} -- Event times (or censoring times.)
        event {np.array[n]} -- Event indicators (0 is censoring).
        surv_func {np.array[n_train, n_test]} -- Survival probabilities n_train x n_test matrix, s.t.
            prob_alive[surv_idx[i], j] gives survial prob at event_time[i] for individual j.
        surv_idx {np.array[n_test]} -- Mapping of survival_func (see surv_func above).

    Returns:
        float -- Time dependent concordance index.
    """
    if surv_func.shape[0] > surv_func.shape[1]:
        warnings.warn(f"consider using 'concordanace_td' when 'surv_func' has more rows than cols.")
    assert event_time.shape[0] == surv_func.shape[1] == surv_idx.shape[0] == event.shape[0]
    assert type(event_time) is type(event) is type(surv_func) is type(surv_idx) is np.ndarray
    if event.dtype in ('float', 'float32'):
        event = event.astype('int32')
    return (_sum_concordant_disc(surv_func, event_time, event, surv_idx) /
            _sum_comparable(event_time, event))


def partial_log_likelihood_ph(log_partial_hazards, durations, events, mean=True):
    """Partial log-likelihood for PH models.
    
    Arguments:
        log_partial_hazards {np.array} -- Log partial hazards (e.g. x^T beta).
        durations {np.array} -- Durations.
        events {np.array} -- Events.
    
    Keyword Arguments:
        mean {bool} -- Return the mean. (default: {True})
    
    Returns:
        pd.Series or float -- partial log-likelihood or mean.
    """

    df = pd.DataFrame(dict(duration=durations, event=events, lph=log_partial_hazards))
    pll = (df
           .sort_values('duration', ascending=False)
           .assign(cum_ph=(lambda x: x['lph']
                            .pipe(np.exp)
                            .cumsum()
                            .groupby(x['duration'])
                            .transform('max')))
           .loc[lambda x: x['event'] == 1]
           .assign(pll=lambda x: x['lph'] - np.log(x['cum_ph']))
           ['pll'])
    if mean:
        return pll.mean()
    return pll
