import numpy as np
from pycox.evaluation import utils

def test_kaplan_meier():
    durations = np.array([1., 1., 2., 3.])
    events = np.array([1, 1, 1, 0])
    surv = utils.kaplan_meier(durations, events)
    assert (surv.index.values == np.arange(4, dtype=float)).all()
    assert (surv.values == np.array([1., 0.5, 0.25, 0.25])).all()
