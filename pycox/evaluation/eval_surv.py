
import warnings
import numpy as np
import pandas as pd
# from pycox.evaluation.inverce_censor_weight import binomial_log_likelihood, brier_score,\
#     integrated_binomial_log_likelihood, integrated_brier_score
from pycox.evaluation.concordance import concordance_td
# from pycox.evaluation.km_inverce_censor_weight import binomial_log_likelihood_km, brier_score_km,\
#     integrated_binomial_log_likelihood_km_numpy, integrated_brier_score_km_numpy
from pycox.evaluation import utils, ipcw


class EvalSurv:
    """Class for evaluating predictions.
    
    Arguments:
        surv {pd.DataFrame} -- Survival preidictions.
        durations {np.array} -- Durations of test set.
        events {np.array} -- Events of test set.

    Keyword Arguments:
        censor_surv {str, pd.DataFrame, EvalSurv} -- Censoring distribution.
            If provided data frame (survival function for censoring) or EvalSurv object,
            this will be used. 
            If 'km', we will fit a Kaplan-Meier to the dataset.
            (default: {None})
    """
    def __init__(self, surv, durations, events, censor_surv=None):
        assert (type(durations) == type(events) == np.ndarray)
        self.surv = surv
        self.durations = durations
        self.events = events
        self.censor_surv = censor_surv
        if type(self.censor_surv) is str:
            if self.censor_surv == 'km':
                self.add_km_censor()
            else:
                raise ValueError(f"censor_surv cannot be {self.censor_surv}.")
        elif self.censor_surv is not None:
            self.add_censor_est(self.censor_surv)
        self.index_surv = self.surv.index.values
        assert pd.Series(self.index_surv).is_monotonic

    def add_censor_est(self, censor_surv):
        """Add censoring estimates so one can use invece censoring weighting.
        `censor_surv` are the suvival estimes trainied on (durations, 1-events),
        
        Arguments:
            censor_surv {pd.DataFrame} -- Censor survival curves.
        """
        if type(censor_surv) is not EvalSurv:
            censor_surv = EvalSurv(censor_surv, self.durations, 1-self.events)
        self.censor_surv = censor_surv
        return self

    def add_km_censor(self):
        """Add censoring estimates obtaind by Kaplan-Meier on the test set
        (durations, 1-events).
        """
        # km = KaplanMeierFitter().fit(self.durations, 1-self.events).survival_function_['KM_estimate']
        # surv = pd.DataFrame(np.repeat(km.values.reshape(-1, 1), len(self.durations), axis=1),
        #                     index=km.index)
        km = utils.kaplan_meier(self.durations, 1-self.events)
        surv = pd.DataFrame(np.repeat(km.values.reshape(-1, 1), len(self.durations), axis=1),
                            index=km.index)
        return self.add_censor_est(surv)

    @property
    def _constructor(self):
        return EvalSurv

    def __getitem__(self, index):
        surv = self.surv.iloc[:, index]
        durations = self.durations[index]
        events = self.events[index]
        censor_surv = self.censor_surv.surv.iloc[:, index] if self.censor_surv is not None else None
        return self._constructor(surv, durations, events, censor_surv)

    def idx_at_times(self, times):
        """Get the index (iloc) of the `surv.index` closest to `times`.
        I.e. surv.loc[tims] (almost)= surv.iloc[idx_at_times(times)].

        Useful for finding predictions at given durations.
        """
        return utils.idx_at_times(self.index_surv, times)

    def _duration_idx(self):
        return self.idx_at_times(self.durations)

    def surv_at_times(self, times):
        idx = self.idx_at_times(times)
        return self.surv.iloc[idx]

    # def prob_alive(self, time_grid):
    #     return self.surv_at_times(time_grid).values

    def concordance_td(self, method='adj_antolini'):
        """Time dependent concorance index from
        Antolini, L.; Boracchi, P.; and Biganzoli, E. 2005. A timedependent discrimination
        index for survival data. Statistics in Medicine 24:3927–3944.

        If 'method' is 'antolini', the concordance from Antolini et al. is computed.
    
        If 'method' is 'adj_antolini' (default) we have made a small modifications
        for ties in predictions and event times.
        We have followed step 3. in Sec 5.1. in Random Survial Forests paper, except for the last
        point with "T_i = T_j, but not both are deaths", as that doesn't make much sense.
        See 'metrics._is_concordant'.

        Keyword Arguments:
            method {str} -- Type of c-index 'antolini' or 'adj_antolini' (default {'adj_antolini'}).

        Returns:
            float -- Time dependent concordance index.
        """
        return concordance_td(self.durations, self.events, self.surv.values,
                              self._duration_idx(), method)

    # def brier_score_km(self, time_grid):
    #     warnings.warn("brier_score_km' will be removed. Use 'add_km_censor' and 'brier_score' instaead.", FutureWarning)
    #     prob_alive = self.prob_alive(time_grid)
    #     bs = brier_score_km(time_grid, prob_alive, self.durations, self.events)
    #     return pd.Series(bs, index=time_grid).rename('brier_score_km')
    
    # def mbll_km(self, time_grid):
    #     warnings.warn("mbll_km' will be removed. Use 'add_km_censor' and 'mbll_km' instaead.", FutureWarning)
    #     prob_alive = self.prob_alive(time_grid)
    #     mbll = binomial_log_likelihood_km(time_grid, prob_alive, self.durations, self.events)
    #     return pd.Series(mbll, index=time_grid).rename('mbll_km')

    # def integrated_brier_score_km(self, time_grid):
    #     warnings.warn("integrated_brier_score_km' will be removed. Use 'add_km_censor' and 'integrated_brier_score' instaead.", FutureWarning)
    #     prob_alive = self.prob_alive(time_grid)
    #     bs = integrated_brier_score_km_numpy(time_grid, prob_alive, self.durations, self.events)
    #     return bs

    # def integrated_mbll_km(self, time_grid):
    #     warnings.warn("integrated_mbll_km' will be removed. Use 'add_km_censor' and 'integrated_mbll_km' instaead.", FutureWarning)
    #     prob_alive = self.prob_alive(time_grid)
    #     score = integrated_binomial_log_likelihood_km_numpy(time_grid, prob_alive, self.durations, self.events)
    #     return score

    def brier_score(self, time_grid, max_weight=np.inf):
        """Brier score weighted by the inverce censoring distibution.
        
        Arguments:
            time_grid {np.array} -- Duarations where the brier score should be calculated.

        Keyword Arguments:
            max_weight {float} -- Max weight value (max number of individuals an individual
                can represent (default {np.inf}).
        """
        if self.censor_surv is None:
            raise ValueError("""Need to add censor_surv to compute briser score. Use 'add_censor_est'
            or 'add_km_censor' for Kaplan-Meier""")
        bs = ipcw.brier_score(time_grid, self.durations, self.events, self.surv.values,
                              self.censor_surv.surv.values, self.index_surv,
                              self.censor_surv.index_surv, max_weight)
        return pd.Series(bs, index=time_grid).rename('brier_score')

    def mbll(self, time_grid, max_weight=np.inf):
        """Mean binomial log-likelihood weighted by the inverce censoring distribution.
        
        Arguments:
            time_grid {np.array} -- Duarations where the brier score should be calculated.

        Keyword Arguments:
            max_weight {float} -- Max weight value (max number of individuals an individual
                can represent (default {np.inf}).
        """
        if self.censor_surv is None:
            raise ValueError("""Need to add censor_surv to compute briser score. Use 'add_censor_est'
            or 'add_km_censor' for Kaplan-Meier""")
        bs = ipcw.binomial_log_likelihood(time_grid, self.durations, self.events, self.surv.values,
                                          self.censor_surv.surv.values, self.index_surv,
                                          self.censor_surv.index_surv, max_weight)
        return pd.Series(bs, index=time_grid).rename('mbll')

    def integrated_brier_score(self, time_grid, max_weight=np.inf):
        """Integrated Brier score weighted by the inverce censoring distibution.
        Essentially an integral over values obtained from `brier_score(time_grid, max_weight)`.
        
        Arguments:
            time_grid {np.array} -- Duarations where the brier score should be calculated.

        Keyword Arguments:
            max_weight {float} -- Max weight value (max number of individuals an individual
                can represent (default {np.inf}).
        """
        if self.censor_surv is None:
            raise ValueError("Need to add censor_surv to compute briser score. Use 'add_censor_est'")
        return ipcw.integrated_brier_score(time_grid, self.durations, self.events, self.surv.values,
                                           self.censor_surv.surv.values, self.index_surv,
                                           self.censor_surv.index_surv, max_weight)

    def integrated_mbll(self, time_grid, max_weight=np.inf):
        """Integrated mean binomial log-likelihood weighted by the inverce censoring distribution.
        Essentially an integral over values obtained from `mbll(time_grid, max_weight)`.
        
        Arguments:
            time_grid {np.array} -- Duarations where the brier score should be calculated.

        Keyword Arguments:
            max_weight {float} -- Max weight value (max number of individuals an individual
                can represent (default {np.inf}).
        """
        if self.censor_surv is None:
            raise ValueError("Need to add censor_surv to compute briser score. Use 'add_censor_est'")
        return ipcw.integrated_binomial_log_likelihood(time_grid, self.durations, self.events, self.surv.values,
                                                       self.censor_surv.surv.values, self.index_surv,
                                                       self.censor_surv.index_surv, max_weight)
