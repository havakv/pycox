'''
Some relevant metrics
'''
import numpy as np
import pandas as pd

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
