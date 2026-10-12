import numpy as np
import pytest
from pycox.evaluation.concordance import concordance_td


def test_concordance_td_mismatched_lengths():
    surv = np.ones((3, 4))
    with pytest.raises(AssertionError, match='got 5, 6, 7 and 4'):
        concordance_td(np.ones(5), np.ones(6), surv, np.zeros(7, dtype='int'))


def test_concordance_td_perfect_ranking():
    durations = np.array([1., 2., 3.])
    events = np.array([1, 1, 1])
    surv = np.array([[0.1, 0.9, 0.95],
                     [0.05, 0.5, 0.9],
                     [0.01, 0.2, 0.5]])
    surv_idx = np.array([0, 1, 2])
    assert concordance_td(durations, events, surv, surv_idx) == 1.
