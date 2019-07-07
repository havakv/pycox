
import warnings
import numpy as np
import pandas as pd
from pycox.evaluation.concordance import concordance_td
from pycox.evaluation import utils, ipcw


class EvalSurv:
    """Class for evaluating predictions.
    
    Arguments:
        surv {pd.DataFrame} -- Survival predictions.
        durations {np.array} -- Durations of test set.
        events {np.array} -- Events of test set.

    Keyword Arguments:
        censor_surv {str, pd.DataFrame, EvalSurv} -- Censoring distribution.
            If provided data frame (survival function for censoring) or EvalSurv object,
            this will be used. 
            If 'km', we will fit a Kaplan-Meier to the dataset.
            (default: {None})
        steps {str} -- For durations between values of `surv.index` choose the higher index 'pre'
            or lower index 'post'. For a visualization see `help(EvalSurv.steps)`. (default: {'pre'})
    """
    def __init__(self, surv, durations, events, censor_surv=None, steps='pre'):
        assert (type(durations) == type(events) == np.ndarray)
        self.surv = surv
        self.durations = durations
        self.events = events
        self.censor_surv = censor_surv
        self.steps = steps
        assert pd.Series(self.index_surv).is_monotonic

    @property
    def censor_surv(self):
        """Estimated survival for censorings. 
        Also an EvalSurv object.
        """
        return self._censor_surv

    @censor_surv.setter
    def censor_surv(self, censor_surv):
        if isinstance(censor_surv, EvalSurv):
            self._censor_surv = censor_surv
        elif type(censor_surv) is str:
            if censor_surv == 'km':
                self.add_km_censor()
            else:
                raise ValueError(f"censor_surv cannot be {censor_surv}. Use e.g. 'km'")
        elif censor_surv is not None:
            self.add_censor_est(censor_surv)
        else:
            self._censor_surv = None

    @property
    def index_surv(self):
        return self.surv.index.values

    @property
    def steps(self):
        """How to handle predictions that are between two indexes in `index_surv`.

        For a visualization, run the following:
            ev = EvalSurv(pd.DataFrame(np.linspace(1, 0, 7)), np.empty(7), np.ones(7), steps='pre')
            ax = ev[0].plot_surv()
            ev.steps = 'post'
            ev[0].plot_surv(ax=ax, style='--')
            ax.legend(['pre', 'post'])
        """
        return self._steps

    @steps.setter
    def steps(self, steps):
        vals = ['post', 'pre']
        if steps not in vals:
            raise ValueError(f"`steps` needs to be {vals}, got {steps}")
        self._steps = steps

    def add_censor_est(self, censor_surv, steps='post'):
        """Add censoring estimates so one can use invece censoring weighting.
        `censor_surv` are the suvival estimes trainied on (durations, 1-events),
        
        Arguments:
            censor_surv {pd.DataFrame} -- Censor survival curves.

    Keyword Arguments:
        round {str} -- For durations between values of `surv.index` choose the higher index 'pre'
            or lower index 'post'. If `None` use `self.steps` (default: {None})
        """
        if not isinstance(censor_surv, EvalSurv):
            censor_surv = self._constructor(censor_surv, self.durations, 1-self.events, None,
                                            steps=steps)
        self.censor_surv = censor_surv
        return self

    def add_km_censor(self, steps='post'):
        """Add censoring estimates obtaind by Kaplan-Meier on the test set
        (durations, 1-events).
        """
        km = utils.kaplan_meier(self.durations, 1-self.events)
        surv = pd.DataFrame(np.repeat(km.values.reshape(-1, 1), len(self.durations), axis=1),
                            index=km.index)
        return self.add_censor_est(surv, steps)

    @property
    def _constructor(self):
        return EvalSurv

    def __getitem__(self, index):
        if not (hasattr(index, '__iter__') or type(index) is slice) :
            index = [index]
        surv = self.surv.iloc[:, index]
        durations = self.durations[index]
        events = self.events[index]
        new = self._constructor(surv, durations, events, None, steps=self.steps)
        if self.censor_surv is not None:
            new.censor_surv = self.censor_surv[index]
        return new

    def plot_surv(self, **kwargs):
        """Plot survival estimates. 
        kwargs are passed to `self.surv.plot`.
        """
        if len(self.durations) > 50:
            raise RuntimeError("We don't allow to plot more than 50 lines. Use e.g. `ev[1:5].plot()`")
        if 'drawstyle' in kwargs:
            raise RuntimeError(f"`drawstyle` is set by `self.steps`. Remove from **kwargs")
        return self.surv.plot(drawstyle=f"steps-{self.steps}", **kwargs)

    def idx_at_times(self, times):
        """Get the index (iloc) of the `surv.index` closest to `times`.
        I.e. surv.loc[tims] (almost)= surv.iloc[idx_at_times(times)].

        Useful for finding predictions at given durations.
        """
        return utils.idx_at_times(self.index_surv, times, self.steps)

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
                              self.censor_surv.index_surv, max_weight, True, self.steps,
                              self.censor_surv.steps)
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
                                          self.censor_surv.index_surv, max_weight, True, self.steps,
                                          self.censor_surv.steps)
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
                                           self.censor_surv.index_surv, max_weight, self.steps,
                                           self.censor_surv.steps)

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
                                                       self.censor_surv.index_surv, max_weight, self.steps,
                                                       self.censor_surv.steps)
