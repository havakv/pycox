import numpy as np
import pandas as pd
from pycox.evaluation import admin
from pycox.evaluation import EvalSurv


def test_brier_score_no_censor():
    n = 4
    durations = np.ones(n) * 50
    durations_c = np.ones_like(durations) * 100
    events = durations <= durations_c
    m = 5
    index_surv = np.array([0, 25., 50., 75., 100.])

    surv_ones = np.ones((m, n))
    time_grid = np.array([5., 40., 60., 100.])
    bs = admin.brier_score(time_grid, durations, durations_c, events, surv_ones, index_surv)
    assert (bs == np.array([0., 0., 1., 1.])).all()
    surv_zeros = surv_ones * 0
    bs = admin.brier_score(time_grid, durations, durations_c, events, surv_zeros, index_surv)
    assert (bs == np.array([1., 1., 0., 0.])).all()
    surv_05 = surv_ones * 0.5
    bs = admin.brier_score(time_grid, durations, durations_c, events, surv_05, index_surv)
    assert (bs == np.array([0.25, 0.25, 0.25, 0.25])).all()
    time_grid = np.array([110.])
    bs = admin.brier_score(time_grid, durations, durations_c, events, surv_05, index_surv)
    assert np.isnan(bs).all()


def test_brier_score_censor():
    n = 4
    durations = np.ones(n) * 50
    durations_c = np.array([25, 50, 60, 100])
    events = durations <= durations_c
    durations[~events] = durations_c[~events]
    m = 5
    index_surv = np.array([0, 25., 50., 75., 100.])

    surv = np.ones((m, n))
    surv[:, 0] = 0
    time_grid = np.array([5., 25., 40., 60., 100.])
    bs = admin.brier_score(time_grid, durations, durations_c, events, surv, index_surv)
    assert (bs == np.array([0.25, 0.25, 0., 1., 1.])).all()


def test_brier_score_evalsurv():
    n = 4
    durations = np.ones(n) * 50
    durations_c = np.array([25, 50, 60, 100])
    events = durations <= durations_c
    durations[~events] = durations_c[~events]
    m = 5
    index_surv = np.array([0, 25., 50., 75., 100.])

    surv = np.ones((m, n))
    surv[:, 0] = 0
    surv = pd.DataFrame(surv, index_surv)
    time_grid = np.array([5., 25., 40., 60., 100.])
    ev = EvalSurv(surv, durations, events, censor_durations=durations_c)
    bs = ev.brier_score_admin(time_grid)
    assert (bs.values == np.array([0.25, 0.25, 0., 1., 1.])).all()
