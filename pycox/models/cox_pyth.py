'''
File contains an implementation of cox regression with arbirary neural network as input.
'''
import warnings
import numpy as np
import pandas as pd

import torch
# from torch import nn, optim

# from ..callbacks.callbacks import CallbacksList, TrainingLogger#, EarlyStoppingTrainLoss
# from ..dataloader import DataLoaderSlice, CoxPrepare, CoxPrepareWithTime, NumpyTensorDataset
from ..metrics import concordance_index, brier_score, integrated_brier_score
# from .base import BaseModel
# from .torch_models import FuncTorch, _Expg

import pyth
from pyth import Model, tuplefy, TupleLeaf

# from torch.utils import data
from pycox.dataloader import sample_alive_from_dates

def make_at_risk_dict(durations):
    '''Create dict(duration: indices) from sorted df.

    Parameters:
        df: A Pandas dataframe with covariates, sorted by duration_col.
        duration_col: Column holding the durations.

    Returns:
        A dict mapping durations to indices (row number, not index in data frame).
        For each time => index of all individual alive.
    '''
    assert type(durations) is np.ndarray, 'Need durations to be a numpy array'
    durations = pd.Series(durations)
    assert durations.is_monotonic_increasing, 'Requires durations to be monotonic'
    allidx = durations.index.values
    keys = durations.drop_duplicates(keep='first')
    at_risk_dict = dict()
    for ix, t in keys.iteritems():
        at_risk_dict[t] = allidx[ix:]
    return at_risk_dict


class CoxPrepare(torch.utils.data.Dataset):
    def __init__(self, input, durations, events, n_control=1):
        df_train_target = pd.DataFrame(dict(duration=durations, event=events))
        self.durations = df_train_target.loc[lambda x: x['event'] == 1]['duration']
        self.at_risk_dict = make_at_risk_dict(durations)

        self.input = tuplefy(input)
        assert type(self.durations) is pd.Series
        self.n_control = n_control

    def __getitem__(self, index):
        if not hasattr(index, '__iter__'):
            index = [index]
        fails = self.durations.iloc[index]
        x_case = self.input.iloc[fails.index]
        control_idx = sample_alive_from_dates(fails.values, self.at_risk_dict, self.n_control)
        x_control = TupleLeaf(self.input.iloc[idx] for idx in control_idx.transpose())
        return tuplefy(x_case, x_control).to_tensor(), None

    def __len__(self):
        return len(self.durations)


class CoxTimePrepare(CoxPrepare):
    def __init__(self, input, durations, events, n_control=1):
        super().__init__(input, durations, events, n_control)
        self.durations_tensor = pyth.tuplefy(self.durations.values.reshape(-1, 1)).to_tensor()

    def __getitem__(self, index):
        if not hasattr(index, '__iter__'):
            index = [index]
        durations = self.durations_tensor.iloc[index]
        (case, control), _ = super().__getitem__(index)
        case = case + durations
        control = control.apply_nrec(lambda x: x + durations)
        return tuplefy(case, control), None


def loss_cox(g_case, g_control, clamp=(-3e+38, 88.)): 
    control_sum = 0.
    for ctr in g_control:
        ctr = ctr - g_case
        ctr = torch.clamp(ctr, *clamp)  # Kills grads for very bad cases (should find better way).
        control_sum += torch.exp(ctr)
    loss = torch.log(1. + control_sum)
    return torch.mean(loss)


class CoxBase(Model):
    def __init__(self, net, optimizer=None, device=None):
        loss = loss_cox
        super().__init__(net, loss, optimizer, device)

    def compute_metrics(self, input, target, metrics):
        if (self.loss is None) and (self.loss in metrics.values()):
            raise RuntimeError(f"Need to specify a loss (self.loss). It's currently None")
        assert target is None, 'Need target to be none, input=(case, control)'
        batch_size = input.lens().flatten().get_if_all_equal()
        if batch_size is None:
            raise RuntimeError("All elements in input does not have the same lenght.")
        case, control = input # both are TupleTree
        input_all = TupleLeaf((case,) + control).cat()
        g_all = self.net(*input_all)
        g_all = tuplefy(g_all).split(batch_size).flatten()
        g_case = g_all[0]
        g_control = g_all[1:]
        res = {name: metric(g_case, g_control) for name, metric in metrics.items()}
        return res

    def make_dataloader_predict(self, input, batch_size, shuffle=False, num_workers=0):
        """Dataloader for prediction. The input is either the regular input, or a tuple
        with input and label.
        
        Arguments:
            input {np.array, tensor, tuple} -- Input to net, or tuple with input and labels.
            batch_size {int} -- Batch size.
        
        Keyword Arguments:
            shuffle {bool} -- If we should shuffle in the dataloader. (default: {False})
            num_workers {int} -- Number of worker in dataloader. (default: {0})
        
        Returns:
            dataloader -- A dataloader.
        """
        dataloader = super().make_dataloader(input, batch_size, shuffle, num_workers)
        return dataloader
    
    def make_dataloader(self, data, batch_size, shuffle=True, num_workers=0, n_control=1):
        """Dataloader for training. Data is on the form (input, target), where
        target is (durations, events).
        
        Arguments:
            data {tuple} -- Tuple containig (input, (durations, events)).
            batch_size {int} -- Batch size.
        
        Keyword Arguments:
            shuffle {bool} -- If shuffle in dataloader (default: {True})
            num_workers {int} -- Number of workers in dataloader. (default: {0})
            n_control {int} -- Number of control samples in dataloader (default: {1})
        
        Returns:
            dataloader -- Dataloader for training.
        """
        input, target = data
        target = tuplefy(target).to_numpy()
        durations, events = target
        idx_sort = np.argsort(durations)
        input = tuplefy(input).iloc[idx_sort]
        durations, events = target.iloc[idx_sort]
        self.df_train_target = pd.DataFrame(dict(duration=durations, event=events))
        self.duration_col = 'duration'
        self.event_col = 'event'
        self.training_data = (input, self.df_train_target)

        dataset = self.make_dataset(input, durations, events, n_control)
        dataloader = pyth.data.DataLoaderSlice(dataset, batch_size=batch_size,
                                               shuffle=shuffle, num_workers=num_workers)
        return dataloader

    make_dataset = NotImplementedError

    def _compute_baseline_hazards(self, input, df_train_target, max_duration, batch_size):
        raise NotImplementedError

    def compute_baseline_hazards(self, training_data=None, max_duration=None, sample=None, batch_size=8224,
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
        if training_data is None:
            if not hasattr(self, 'training_data'):
                raise ValueError('Need to fit, or supply a training_data to this function.')
            training_data = self.training_data
        
        input, df_train_target = training_data
        df_train_target = df_train_target.reset_index(drop=True)
        
        if sample is not None:
            if sample >= 1:
                df_train_target = df_train_target.sample(n=sample)
            else:
                df_train_target = df_train_target.sample(frac=sample)
            input = input.iloc[df_train_target.index.values]
        
        base_haz = self._compute_baseline_hazards(input, df_train_target, max_duration, batch_size)

        if set_hazards:
            self.compute_baseline_cumulative_hazards(set_hazards=True, baseline_hazards_=base_haz)

        return base_haz

    def compute_baseline_cumulative_hazards(self, training_data=None, max_duration=None, sample=None,
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
        if (training_data is not None) and (baseline_hazards_ is not None):
            raise ValueError("'training_data' and 'baseline_hazards_' can not both be different from 'None'.")
        if baseline_hazards_ is None:
            baseline_hazards_ = self.compute_baseline_hazards(training_data, max_duration, sample, batch_size,
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

    def predict_cumulative_hazards(self, input, max_duration=None, batch_size=16448, verbose=False, baseline_hazards_=None):
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

    def predict_survival_function(self, input, max_duration=None, batch_size=512, verbose=False, baseline_hazards_=None):
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

    def predict_cumulative_hazards_at_times(self, times, input, batch_size=16448, return_df=True,
                                            verbose=False, baseline_hazards_=None):
        raise NotImplementedError

    def predict_survival_at_times(self, times, input, batch_size=16448, return_df=True,
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

    def brier_score(self, times, input, target, batch_size=512):
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
                               batch_size=512):
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


class CoxPH(CoxBase):
    make_dataset = CoxPrepare

    def _compute_baseline_hazards(self, input, df_train_target, max_duration, batch_size):
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
        return (df_train_target
                .assign(expg=np.exp(self.predict(input, batch_size)))
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
        expg = np.exp(self.predict(input, batch_size)).reshape(1, -1)
        return pd.DataFrame(bch.values.reshape(-1, 1).dot(expg), 
                            index=bch.index)

    def predict_cumulative_hazards_at_times(self, times, input, batch_size=16448, return_df=True,
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
            warnings.warn('No verbose to show...')
        if baseline_hazards_ is None:
            bch = self.baseline_cumulative_hazards_
        else:
            bch = self.compute_baseline_cumulative_hazards(batch_size=batch_size, set_hazards=False,
                                                           baseline_hazards_=baseline_hazards_)
        if not hasattr(times, '__iter__'):
            times = [times]
        times_idx = search_sorted_idx(bch.index.values, times)
        bch = bch.iloc[times_idx].values.reshape(-1, 1)
        expg = np.exp(self.predict(input, batch_size)).reshape(1, -1)
        res = bch.dot(expg)
        if return_df:
            return pd.DataFrame(res, index=times)
        return res

    def partial_log_likelihood(self, input, target, g_preds=None, batch_size=512):
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
        durations, events = target
        df = pd.DataFrame({self.duration_col: durations, self.event_col: events})
        if g_preds is None:
            g_preds = self.predict(input, batch_size)
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
            g_preds = self.predict(input, batch_size).flatten()
        return 1 - concordance_index(durations, g_preds, events)


class CoxTime(CoxBase):
    make_dataset = CoxTimePrepare

    def make_dataloader_predict(self, input, batch_size, shuffle=False, num_workers=0):
        input, durations = input
        input = tuplefy(input)
        durations = tuplefy(durations)
        new_input = input + durations 
        dataloader = super().make_dataloader_predict(new_input, batch_size, shuffle, num_workers)
        return dataloader

    def _compute_baseline_hazards(self, input, df_train_target, max_duration, batch_size):
        '''Computes the breslow estimates of the baseline hazards of dataframe df.

        Parameters:
            df: Pandas dataframe with covariates, duration, and events.
            max_duration: Don't compute hazards for durations larger than max_time.
            batch_size: Batch size passed calculation of g_preds.

        Returns:
            Pandas series with baseline hazards. Index is duration_col.
        '''
        if max_duration is None:
            max_duration = np.inf
        def compute_expg_at_risk(ix, t):
            sub = input.iloc[ix:]
            n = sub.lens().flatten().get_if_all_equal()
            t = np.repeat(t, n).reshape(-1, 1).astype('float32')
            return np.exp(self.predict((sub, t), batch_size)).flatten().sum()

        if not df_train_target[self.duration_col].is_monotonic_increasing:
            raise RuntimeError(f"Need 'df_train_target' to be sorted by {self.duration_col}")
        input = tuplefy(input)
        df = df_train_target.reset_index(drop=True)
        times = (df
                 .loc[lambda x: x[self.event_col] != 0]
                 [self.duration_col]
                 .loc[lambda x: x <= max_duration]
                 .drop_duplicates(keep='first'))
        at_risk_sum = (pd.Series([compute_expg_at_risk(ix, t) for ix, t in times.iteritems()],
                                 index=times.values)
                       .rename('at_risk_sum'))
        events = (df
                  .groupby(self.duration_col)
                  [[self.event_col]]
                  .agg('sum')
                  .loc[lambda x: x.index <= max_duration])
        base_haz =  (events
                     .join(at_risk_sum, how='left', sort=True)
                     .pipe(lambda x: x[self.event_col] / x['at_risk_sum'])
                     .fillna(0.)
                     .rename('baseline_hazards'))
        return base_haz

    def _predict_cumulative_hazards(self, input, max_duration, batch_size, verbose, baseline_hazards_):
        '''Get cumulative hazards for dataset df.
        H(x, t) = sum [h0(t) exp(g(x, t))]

        Parameters:
            df: Pandas dataframe with covariates.
            batch_size: Batch size passed calculation of g_preds.
            verbose: If we should print progress.
            baseline_hazards_: Pandas series with index: time, and values: baseline hazards.

        Returns:
            Pandas data frame with cumulative hazards. One columns for
            each individual in the df.
        '''
        def expg_at_time(t):
            t = np.repeat(t, n_cols).reshape(-1, 1).astype('float32')
            return np.exp(self.predict((input, t), batch_size)).flatten()

        input = tuplefy(input)
        max_duration = np.inf if max_duration is None else max_duration
        baseline_hazards_ = baseline_hazards_.loc[lambda x: x.index <= max_duration]
        n_rows, n_cols = baseline_hazards_.shape[0], input.lens().flatten().get_if_all_equal()
        hazards = np.empty((n_rows, n_cols))
        for idx, t in enumerate(baseline_hazards_.index):
            if verbose:
                print(idx, 'of', len(baseline_hazards_))
            hazards[idx, :] = expg_at_time(t)
        hazards *= baseline_hazards_.values.reshape(-1, 1)
        return pd.DataFrame(hazards, index=baseline_hazards_.index).cumsum()

    def predict_cumulative_hazards_at_times(self, times, input, batch_size=16448, return_df=True,
                                           verbose=False, baseline_hazards_=None):
        '''Predict cumulative hazards only at given times. This is not as efficient as
        for the proportional hazards models.

        Parameters:
            times: Number or iterable with times.
            df: Pandas dataframe with covariates.
            batch_size: Batch size passed calculation of g_preds.
            return_df: Whether or not to return a pandas dataframe or a numpy matrix.
            verbose: If we should print progress.
            baseline_hazards_: Pandas series with index: time, and values: baseline hazards.

        Returns:
            Pandas dataframe (or numpy matrix) [len(times), len(df)] with cumulative hazards
            estimates.
        '''
        if not hasattr(times, '__iter__'):
            times = [times]
        max_duration = max(times)
        cum_haz = self.predict_cumulative_hazards(input, max_duration, batch_size,
                                                  verbose, baseline_hazards_)
        times_idx = search_sorted_idx(cum_haz.index.values, times)
        cum_haz = cum_haz.iloc[times_idx]
        if return_df:
            return cum_haz
        return cum_haz.as_matrix()

    def partial_log_likelihood(self, input, target, batch_size=512):
        '''Calculate the partial log-likelihood for the events in datafram df.
        This likelihood does not sample the controls.
        Note that censored data (non events) does not have a partial log-likelihood.

        Parameters:
            df: Pandas dataframe with covariates, duration, and events.
            batch_size: Batch size passed calculation of g_preds.

        Returns:
            Pandas series with partial likelihood.
        '''
        def expg_sum(t, i):
            sub = input_sorted.iloc[i:]
            n = sub.lens().flatten().get_if_all_equal()
            t = np.repeat(t, n).reshape(-1, 1).astype('float32')
            return np.exp(self.predict((sub, t), batch_size)).flatten().sum()

        durations, events = target
        df = pd.DataFrame({self.duration_col: durations, self.event_col: events})
        df = df.sort_values(self.duration_col)
        input = tuplefy(input)
        input_sorted = input.iloc[df.index.values]

        times =  (df
                  .assign(_idx=np.arange(len(df)))
                  .loc[lambda x: x[self.event_col] == True]
                  .drop_duplicates(self.duration_col, keep='first')
                  .assign(_expg_sum=lambda x: [expg_sum(t, i) for t, i in zip(x[self.duration_col], x['_idx'])])
                  .drop([self.event_col, '_idx'], axis=1))
        
        idx_name_old = df.index.name
        idx_name = '__' + idx_name_old if idx_name_old else '__index'
        df.index.name = idx_name

        pll = df.loc[lambda x: x[self.event_col] == True]
        input_event = input.iloc[pll.index.values]
        durations_event = pll[self.duration_col].values.reshape(-1, 1)
        g_preds = self.predict((input_event, durations_event), batch_size).flatten()
        pll = (pll
               .assign(_g_preds=g_preds)
               .reset_index()
               .merge(times, on=self.duration_col)
               .set_index(idx_name)
               .assign(pll=lambda x: x['_g_preds'] - np.log(x['_expg_sum']))
               ['pll'])

        pll.index.name = idx_name_old
        return pll


def search_sorted_idx(array, values):
    '''For sorted array, get index of values.
    If value not in array, give left index of value.
    '''
    n = len(array)
    idx = np.searchsorted(array, values)
    idx[idx == n] = n-1 # We can't have indexes higher than the length-1
    not_exact = values != array[idx]
    idx -= not_exact
    if any(idx < 0):
        warnings.warn('Given value smaller than first value')
        idx[idx < 0] = 0
    return idx
