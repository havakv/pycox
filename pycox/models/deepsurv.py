"""An implementation of the DeepSurv paper"""

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from pyth import Model, tuplefy, make_dataloader
from pyth.data import DatasetTuple
from pycox.metrics import brier_score, integrated_brier_score
from pycox.models.cox_pyth import search_sorted_idx
# from pycox.models.cox_pyth import CoxPH

def loss_deepsurv(risk, event):
    event = event.view(-1)
    risk = risk.view(-1)
    log_risk = risk.exp().cumsum(0).log()
    return - risk.sub(log_risk).mul(event).sum().div(event.sum())

class DatasetDurationSorted(DatasetTuple):
    def __getitem__(self, index):
        batch = super().__getitem__(index)
        input, (duration, event) = batch
        idx_sort = duration.sort(descending=True)[1]
        event = event.float()
        batch = tuplefy(input, event).iloc[idx_sort]
        return batch

class CoxBase(Model):
    duration_col = 'duration'
    event_col = 'event'

    def _compute_baseline_hazards(self, input, df, max_duration, batch_size):
        raise NotImplementedError

    def target_to_df(self, target):
        durations, events = tuplefy(target).to_numpy()
        df = pd.DataFrame({self.duration_col: durations, self.event_col: events}) 
        return df

    def compute_baseline_hazards(self, input, target, max_duration=None, sample=None, batch_size=8224,
                                set_hazards=True):
        '''Computes the breslow estimates of the baseline hazards of dataframe df.

        Parameters:
            # df: Pandas dataframe with covariates, duration, and events.
            #     If None: use training data frame.
            max_duration: Don't compute hazards for durations larger than max_time.
            sample: Use sample of df. 
                Sample proportion if 'sample' < 1, else sample number 'sample'.
            batch_size: Batch size passed calculation of g_preds.
            set_hazards: If we should store computed hazards in object.

        Returns:
            Pandas series with baseline hazards. Index is duration_col.
        '''
        df = self.target_to_df(target)#.sort_values(self.duration_col)
        if sample is not None:
            if sample >= 1:
                df = df.sample(n=sample)
            else:
                df = df.sample(frac=sample)
        input = tuplefy(input).to_numpy().iloc[df.index.values]
        base_haz = self._compute_baseline_hazards(input, df, max_duration, batch_size)
        if set_hazards:
            self.compute_baseline_cumulative_hazards(set_hazards=True, baseline_hazards_=base_haz)
        return base_haz

    def compute_baseline_cumulative_hazards(self, input=None, target=None, max_duration=None, sample=None,
                                            batch_size=8224, set_hazards=True, baseline_hazards_=None):
        '''Compute the baseline cumulative hazards of dataframe df or baseline_hazards.

        Parameters:
            # df: Pandas dataframe with covariates, duration, and events.
            #     If None: use training data frame.
            max_duration: Don't compute hazards for durations larger than max_time.
            sample: Use sample of df. 
                Sample proportion if 'sample' < 1, else sample number 'sample'.
            batch_size: Batch size passed calculation of g_preds.
            set_hazards: If we should store computed hazards in object.
            baseline_hazards: Pandas series with baseline hazards.
                If `None` use supplied df or training data frame.

        Returns:
            Pandas series with baseline cumulative hazards. Index is duration_col.
        '''
        if ((input is not None) or (target is not None)) and (baseline_hazards_ is not None):
            raise ValueError("'input', 'target' and 'baseline_hazards_' can not both be different from 'None'.")
        if baseline_hazards_ is None:
            baseline_hazards_ = self.compute_baseline_hazards(input, target, max_duration, sample, batch_size,
                                                             set_hazards=False)
        assert baseline_hazards_.index.is_monotonic_increasing,\
            'Need index of baseline_hazards_ to be monotonic increasing, as it represents time.'
        bch = (baseline_hazards_
                .cumsum()
                .rename('baseline_cumulative_hazards'))
        if set_hazards:
            self.baseline_hazards_ = baseline_hazards_
            self.baseline_cumulative_hazards_ = bch
        return bch

    def predict_cumulative_hazards(self, input, max_duration=None, batch_size=8224, verbose=False, baseline_hazards_=None):
        '''Get cumulative hazards for dataset df.
        H(x, t) = sum [h0(t) exp(g(x, t))]
        or
        H(x, t) = sum [h0(t) exp(g(x))]

        Parameters:
            # df: Pandas dataframe with covariates.
            max_duration: Don't compute hazards for durations larger than max_time.
            batch_size: Batch size passed calculation of g_preds.
            verbose: If we should print progress.
            baseline_hazards_: Pandas series with index: time, and values: baseline hazards.
                If None, use baseline_hazards_ of model.

        Returns:
            Pandas data frame with cumulative hazards. One columns for
            each individual in the df.
        '''
        if baseline_hazards_ is None:
            if not hasattr(self, 'baseline_hazards_'):
                raise ValueError('Need to compute baseline_hazards_.')
            baseline_hazards_ = self.baseline_hazards_
        assert baseline_hazards_.index.is_monotonic_increasing,\
            'Need index of baseline_hazards_ to be monotonic increasing, as it represents time.'
        return self._predict_cumulative_hazards(input, max_duration, batch_size, verbose, baseline_hazards_)

    def _predict_cumulative_hazards(self, input, max_duration, batch_size, verbose, baseline_hazards_):
        raise NotImplementedError

    def predict_survival_function(self, input, max_duration=None, batch_size=8224, verbose=False, baseline_hazards_=None):
        '''Predict survival function for dataset df.
        S(x, t) = exp(-H(x, t))

        Parameters:
            # df: Pandas dataframe with covariates.
            max_duration: Don't compute hazards for durations larger than max_time.
            batch_size: Batch size passed calculation of g_preds.
            verbose: If we should print progress.
            baseline_hazards_: Pandas series with index: time, and values: baseline hazards.

        Returns:
            Pandas data frame with survival functions. One columns for
            each individual in the df.
        '''
        return np.exp(-self.predict_cumulative_hazards(input, max_duration, batch_size, verbose, baseline_hazards_))

    def predict_cumulative_hazards_at_times(self, times, input, batch_size=8224, return_df=True,
                                            verbose=False, baseline_hazards_=None):
        raise NotImplementedError

    def predict_survival_at_times(self, times, input, batch_size=8224, return_df=True,
                                  verbose=False, baseline_hazards_=None):
        '''Predict survival function at given times.
        Not very efficient!!!

        Parameters:
            times: Iterable with times.
            df: Pandas dataframe with covariates.
            batch_size: Batch size passed calculation of g_preds.
            return_df: Whether or not to return a pandas dataframe or a numpy matrix.
            verbose: If we should print progress.
            baseline_hazards_: Pandas series with index: time, and values: baseline hazards.

        Returns:
            Pandas dataframe (or numpy matrix) [len(times), len(df)] with survival estimates.
        '''
        return np.exp(-self.predict_cumulative_hazards_at_times(times, input, batch_size, return_df,
                                                                verbose, baseline_hazards_))

    def brier_score(self, times, input, target, batch_size=8224):
        '''Gives brier scores for `times`.

        Parameters:
            times: Number or iterable with times where to compute the score.
            df: Pandas dataframe with covariates, events, and durations.
            batch_size: Batch size passed calculation of g_preds.

        Returns:
            Numpy array with brier scores.
        '''
        prob_alive = self.predict_survival_at_times(times, input, batch_size, False)
        durations, events = target
        return brier_score(times, prob_alive, durations, events)

    def integrated_brier_score(self, input, target, times_grid=None, n_grid_points=100,
                               batch_size=8224):
        '''Compute the integrated brier score (for survival) of df.

        For a specification on brier scores for survival data see e.g.:
        "Assessment of evaluation criteria for survival prediction from
        genomic data" by Bovelstad and Borgan.

        Parameters:
            df: Pandas dataframe with covariates, events, and durations.
            times_grid: Specified time grid for integration. If None: use equidistant between
                smallest and largest value times of durations.
            n_grid_points: Only apply if grid is None. Gives number of grid poinst used
                in equidistant grid.
            batch_size: Batch size passed calculation of g_preds.
        '''
        def prob_alive_func(times):
            return self.predict_survival_at_times(times, input, batch_size=batch_size, return_df=False)

        durations, events = target
        return integrated_brier_score(prob_alive_func, durations, events, times_grid, n_grid_points)


class CoxPHBase(CoxBase):
    def _compute_baseline_hazards(self, input, df_target, max_duration, batch_size):
        '''Computes the breslow estimates of the baseline hazards of dataframe df.

        Parameters:
            df: Pandas dataframe with covariates, duration, and events.
            max_duration: Has no computational effect here.
            batch_size: Batch size passed calculation of g_preds.

        Returns:
            Pandas series with baseline hazards. Index is duration_col.
        '''
        if max_duration is None:
            max_duration = np.inf

        # Here we are computing when expg when there are no events.
        #   Could be made faster, by only computing when there are events.
        return (df_target
                .assign(expg=np.exp(self.predict(input, batch_size, return_numpy=True)))
                .groupby(self.duration_col)
                .agg({'expg': 'sum', self.event_col: 'sum'})
                .sort_index(ascending=False)
                .assign(expg=lambda x: x['expg'].cumsum())
                .pipe(lambda x: x[self.event_col]/x['expg'])
                .fillna(0.)
                .iloc[::-1]
                .loc[lambda x: x.index <= max_duration]
                .rename('baseline_hazards'))

    def _predict_cumulative_hazards(self, input, max_duration, batch_size, verbose, baseline_hazards_):
        '''Get cumulative hazards for dataset df.
        H(x, t) = H0(t) exp(g(x))

        Parameters:
            df: Pandas dataframe with covariates.
            batch_size: Batch size passed calculation of g_preds.

        Returns:
            Pandas data frame with cumulative hazards. One columns for
            each individual in the df.
        '''
        max_duration = np.inf if max_duration is None else max_duration
        if baseline_hazards_ is self.baseline_hazards_:
            bch = self.baseline_cumulative_hazards_
        else:
            bch = self.compute_baseline_cumulative_hazards(set_hazards=False, 
                                                           baseline_hazards_=baseline_hazards_)
        bch = bch.loc[lambda x: x.index <= max_duration]
        expg = np.exp(self.predict(input, batch_size, return_numpy=True)).reshape(1, -1)
        return pd.DataFrame(bch.values.reshape(-1, 1).dot(expg), 
                            index=bch.index)

    def predict_cumulative_hazards_at_times(self, times, input, batch_size=8224, return_df=True,
                                            verbose=False, baseline_hazards_=None):
        '''Predict cumulative hazards H(x, t) = exp(- H0(t)*exp(g(x))), only at given times.

        Parameters:
            times: Number or iterable with times.
            df: Pandas dataframe with covariates.
            batch_size: Batch size passed calculation of g_preds.
            return_df: Whether or not to return a pandas dataframe or a numpy matrix.

        Returns:
            Pandas dataframe (or numpy matrix) [len(times), len(df)] with cumulative hazards
            estimates.
        '''
        if verbose:
            print('No verbose to show...')
        if baseline_hazards_ is None:
            bch = self.baseline_cumulative_hazards_
        else:
            bch = self.compute_baseline_cumulative_hazards(batch_size=batch_size, set_hazards=False,
                                                           baseline_hazards_=baseline_hazards_)
        if not hasattr(times, '__iter__'):
            times = [times]
        times_idx = search_sorted_idx(bch.index.values, times)
        bch = bch.iloc[times_idx].values.reshape(-1, 1)
        expg = np.exp(self.predict(input, batch_size, return_numpy=True)).reshape(1, -1)
        res = bch.dot(expg)
        if return_df:
            return pd.DataFrame(res, index=times)
        return res

    def partial_log_likelihood(self, input, target, g_preds=None, batch_size=8224):
        '''Calculate the partial log-likelihood for the events in datafram df.
        This likelihood does not sample the controls.
        Note that censored data (non events) does not have a partial log-likelihood.

        Parameters:
            df: Pandas dataframe with covariates, duration, and events.
            g_preds: Exponent of proportional hazards (h = h0 * exp(g(x))).
                If not supplied, it will be calculated.
            batch_size: Batch size passed calculation of g_preds.

        Returns:
            Pandas dataframe with duration, g_preds, and the
                partial log-likelihood pll.
        '''
        df = self.target_to_df(target)
        if g_preds is None:
            g_preds = self.predict(input, batch_size, return_numpy=True)
        return (df
                .assign(_g_preds=g_preds)
                .sort_values(self.duration_col, ascending=False)
                .assign(_cum_exp_g=(lambda x: x['_g_preds']
                                    .pipe(np.exp)
                                    .cumsum()
                                    .groupby(x[self.duration_col])
                                    .transform('max')))
                .loc[lambda x: x[self.event_col] == 1]
                .assign(pll=lambda x: x['_g_preds'] - np.log(x['_cum_exp_g']))
                ['pll'])

    def concordance_index(self, input, target, g_preds=None, batch_size=256):
        '''Concoradance index (from lifelines.utils).
        If g_preds are not supplied (None), they will be calculated.
            h = h0 * exp(g(x)).

        Parameters:
            df: Pandas dataframe with covariates, duration, and events.
            g_preds: Exponent of proportional hazards (h = h0 * exp(g(x))).
                If not supplied, it will be calculated.
            batch_size: Batch size passed calculation of g_preds.
        '''
        durations, events = target
        if g_preds is None:
            g_preds = self.predict(input, batch_size, return_numpy=True).flatten()
        return 1 - concordance_index(durations, g_preds, events)


class DeepSurv(CoxPHBase):
    def __init__(self, net, optimizer=None, device=None):
        loss = loss_deepsurv
        return super().__init__(net, loss=loss, optimizer=optimizer, device=device)

    @staticmethod
    def make_dataloader(data, batch_size, shuffle, num_workers=0):
        dataloader = make_dataloader(data, batch_size, shuffle, num_workers,
                                     make_dataset=DatasetDurationSorted)
        return dataloader

    def make_dataloader_predict(self, input, batch_size, shuffle=False, num_workers=0):
        dataloader = super().make_dataloader(input, batch_size, shuffle, num_workers)
        return dataloader
