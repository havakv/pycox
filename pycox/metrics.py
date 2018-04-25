'''
Some relevant metrics
'''
import warnings

import numpy as np
import scipy
import pandas as pd

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
