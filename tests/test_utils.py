import pytest
import numpy as np
from pycox import utils

def test_kaplan_meier():
    durations = np.array([1., 1., 2., 3.])
    events = np.array([1, 1, 1, 0])
    surv = utils.kaplan_meier(durations, events)
    assert (surv.index.values == np.arange(4, dtype=float)).all()
    assert (surv.values == np.array([1., 0.5, 0.25, 0.25])).all()

@pytest.mark.parametrize('n', [10, 85, 259])
@pytest.mark.parametrize('p_cens', [0, 0.3, 0.8])
def test_kaplan_meier_vs_lifelines(n, p_cens):
    from lifelines import KaplanMeierFitter
    np.random.seed(0)
    durations = np.random.uniform(0, 100, n)
    events = np.random.binomial(1, 1 - p_cens, n).astype('float')
    km = utils.kaplan_meier(durations, events)
    kmf = KaplanMeierFitter().fit(durations, events).survival_function_['KM_estimate']
    assert km.shape == kmf.shape
    assert (km - kmf).abs().max() < 1e-14
    assert (km.index == kmf.index).all()
