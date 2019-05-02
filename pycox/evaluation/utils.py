import pandas as pd
import numpy as np

def idx_at_times(index_surv, times, assert_sorted=True):
    if assert_sorted:
        assert pd.Series(index_surv).is_monotonic_increasing, "Need 'index_surv' to be monotonic increasing"
    idx = np.searchsorted(index_surv, times)
    return idx.clip(0, len(index_surv)-1)