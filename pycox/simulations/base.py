import numpy as np
import pandas as pd

def dict2df(data, add_true=True, add_censor_covs=False):
    """Make a pd.DataFrame from the dict obtained when simulating.

    Arguments:
        data {dict} -- Dict from simulation.

    Keyword Arguments:
        add_true {bool} -- If we should include the true duration and censoring times
            (default: {True})
        add_censor_covs {bool} -- If we should include the censor covariates as covariates.
            (default: {False})

    Returns:
        pd.DataFrame -- A DataFrame
    """
    covs = data['covs']
    if add_censor_covs:
        covs = np.concatenate([covs, data['censor_covs']], axis=1)
    df = (pd.DataFrame(covs, columns=[f"x{i}" for i in range(covs.shape[1])])
          .assign(duration=data['durations'].astype('float32'),
                  event=data['events'].astype('float32')))
    if add_true:
        df = df.assign(duration_true=data['durations_true'].astype('float32'),
                       event_true=data['events_true'].astype('float32'),
                       censoring_true=data['censor_durations'].astype('float32'))
    return df


class _SimBase:
    def simulate(self, n, surv_df=False):
        """Simulate dataset of size `n`.
        
        Arguments:
            n {int} -- Number of simulations
        
        Keyword Arguments:
            surv_df {bool} -- If a dataframe containing the survival function should be returned.
                (default: {False})
        
        Returns:
            [dict] -- A dictionary with the results.
        """
        raise NotImplementedError

    def surv_df(self, *args):
        """Returns a data frame containing the survival function.
        """
        raise NotImplementedError
