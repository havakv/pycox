'''
File contains an implementation of cox regression with arbirary neural network as input.
'''
import warnings
import numpy as np
import pandas as pd

import torch
from torch import nn, optim

from ..callbacks.callbacks import CallbacksList, TrainingLogger#, EarlyStoppingTrainLoss
from ..dataloader import DataLoaderSlice, CoxPrepare, CoxPrepareWithTime, NumpyTensorDataset
from ..metrics import concordance_index, brier_score, integrated_brier_score
from .base import BaseModel
from .torch_models import FuncTorch, _Expg

from torch.utils import data
from pycox.dataloader import sample_alive_from_dates


class CoxBase(BaseModel):
    '''Base class for cox models.

    Parameters:
        g: Torch model for computing g(X). h = h0 exp(g(X)).
            This is equivalent to self.net.
        optimizer: Torch optimizer. If None SGD(lr=0.1, momentum=0.9)
        device: Which device to compute on.
            Preferrably pass a torch.device object.
            If `None`: use default gpu if avaiable, else use cpu.
            If `int`: used that gpu: torch.device('cuda:<device>').
            If `string`: string is passed to torch.device(`string`).
    '''
    def __init__(self, g, optimizer=None, device=None):
        super().__init__(g, optimizer, device)
        self.g = g # This is the same as self.net

    def _is_repeated_fit_df(self, df, duration_col, event_col):
        if hasattr(self, 'df'):
            if (self._df is df) and (self.duration_col == duration_col) and (self.event_col == event_col):
                return True
        return False

    @staticmethod
    def make_at_risk_dict(df, duration_col):
        '''Create dict(duration: indices) from sorted df.

        Parameters:
            df: A Pandas dataframe with covariates, sorted by duration_col.
            duration_col: Column holding the durations.

        Returns:
            A dict mapping durations to indices (row number, not index in data frame).
            For each time => index of all individual alive.
        '''
        assert df[duration_col].is_monotonic_increasing, 'Requires df sorted by `duration_col`.'
        df = df.reset_index(drop=True)
        allidx = df.index.values
        keys = df[duration_col].drop_duplicates(keep='first')

        at_risk_dict = dict()
        for ix, t in keys.iteritems():
            at_risk_dict[t] = allidx[ix:]
        return at_risk_dict

    def _prepare_data_fit(self):
        '''Prepare data for fit.
        Add durations, at_risk_dict, x_columns, and Xtr to self.
        '''
        self.durations = self.df.loc[lambda x: x[self.event_col] == 1][self.duration_col]
        self.at_risk_dict = self.make_at_risk_dict(self.df, self.duration_col)
        Xtr = self.df.drop([self.duration_col, self.event_col, self.old_index_name], axis=1)
        self.x_columns = Xtr.columns
        self.Xtr = Xtr.as_matrix().astype('float32')

    @staticmethod
    def _reset_idx_with_old_idx_name(df):
        '''Reset index of data frame, and return df with
        index column name.
        '''
        cols = set(df.columns)
        df = df.reset_index()
        col = list(set(df.columns) - cols)[0]
        return df, col

    def fit(self, df, duration_col, event_col=None, n_control=1, batch_size=64, epochs=1,
            num_workers=0, verbose=1, strata=None, callbacks=None, compute_hazards=True):
        '''Fit the Cox Propertional Hazards model to a dataset. Tied survival times
        are handled using Beslow's tie-method.

        Parameters:
            df: A Pandas dataframe with necessary columns `duration_col` and
                `event_col`, plus other covariates. `duration_col` refers to
                the lifetimes of the subjects. `event_col` refers to whether
                the 'death' events was observed: 1 if observed, 0 else (censored).
            duration_col: The column in dataframe that contains the subjects'
                lifetimes.
            event_col: The column in dataframe that contains the subjects' death
                observation. If left as None, assume all individuals are non-censored.
            n_control: Number of control samples.
            batch_size: Batch size.
            epochs: Number of epochs.
            num_workers: Number of workers for preparing data.
            verbose: Degree of verbose. If dict {'name': mm}, where mm is a MonitorMetric object,
                this will be printed. 
                Example: 
                mm = MonitorCoxLoss(df_val, n_control=1, n_reps=4,)
                cox.fit(..., verbose={'val_loss': mm}, callbacks=[mm])
            strata: Specify a list of columns to use in stratification. This is useful if a
                catagorical covariate does not obey the proportional hazards assumption. This
                is used similar to the `strata` expression in R.
                See http://courses.washington.edu/b515/l17.pdf.
            callbacks: List of callbacks.
            compute_hazards: If we should compute hazards when training has finished.

        # Returns:
        #     self, with additional properties: hazards_
        '''
        if strata is not None:
            raise NotImplementedError('strata not implemented yet.')
        if event_col is None:
            raise NotImplementedError('need to specify event_col')

        self._repeated_df = self._is_repeated_fit_df(df, duration_col, event_col)
        self.duration_col = duration_col
        self.event_col = event_col
        self._df = df

        # When fitting, we need to sort df by event time (for computational efficiency,
        # and the index of the dataframe to be reset.
        # This is because we remove cencored individuals, and need a continuous
        # index to map between a data frame and a numpy matrix.
        # cols = set(df.columns)
        self.df, self.old_index_name = (df.sort_values(self.duration_col)
                                        .pipe(self._reset_idx_with_old_idx_name))
        if not self._repeated_df:
            self._prepare_data_fit()
        log = self.fit_data(self.Xtr, self.durations, self.at_risk_dict, n_control,
                             batch_size, epochs, num_workers, verbose, callbacks)

        # warnings.warn('unsure about set_hazards')
        # self.set_hazards(self.df, compute_hazards=compute_hazards)
        if compute_hazards:
            self.compute_baseline_hazards()
        return log

    def fit_data(self, data, durations, at_risk_dict, n_control=1,
            batch_size=64, epochs=1, num_workers=0, verbose=1, callbacks=None):
        '''Use fit_dataloader of you prefer to give a DataLoader instead.

        Parametes:
            data: typically a numpy matrix. Passed to self.make_dataloader
            # X: Numpy array with covariates.
            durations: Pandas series with index same as fails in X, and values
                giving time of that failure.
            at_risk_dict: Dict with gr_alive[time] = <array with index of alive in X matrix>.
            n_control: Number of control samples.
            batch_size: Batch size.
            epochs: Number of epochs.
            num_workers: Number of workers for preparing data.
            verbose: Degree of verbose. If dict {'name': mm}, where mm is a MonitorMetric object,
                this will be printed. 
                Example: 
                mm = MonitorCoxLoss(df_val, n_control=1, n_reps=4,)
                cox.fit(..., verbose={'val_loss': mm}, callbacks=[mm])
            callbacks: List of callbacks.
        '''
        dataloader = self.make_dataloader(data, durations, at_risk_dict, n_control, batch_size,
                                          num_workers)
        log = self.fit_dataloader(dataloader, epochs, verbose, callbacks)
        return log

    @staticmethod
    def make_dataloader(X, durations, at_risk_dict, n_control, batch_size, num_workers):
        raise NotImplementedError

    def fit_dataloader(self, dataloader, epochs=1, verbose=1, callbacks=None):
        '''Fit method for pytorch dataloader.'''
        self._setup_train_info(dataloader, verbose, callbacks)

        stop_signal = self.callbacks.on_fit_start()
        if stop_signal:
            raise RuntimeError('Got stop_signal from callback before fit starts')
        for _ in range(epochs):
            for case, control in dataloader:
                self.fit_info['case_control'] = (case, control)
                case, control = case.to(self.device), control.to(self.device)
                g_case, g_control = self.compute_g_case_control(case, control)
                self.fit_info['g_case_control'] = (g_case, g_control)
                self.batch_loss = self.loss_func(g_case, g_control)
                self.optimizer.zero_grad()
                self.batch_loss.backward()
                stop_signal = self.callbacks.before_step()
                if stop_signal:
                    raise RuntimeError('Stop signal in before_step().')
                self.optimizer.step()
                stop_signal += self.callbacks.on_batch_end()
                if stop_signal:
                    break
            else:
                stop_signal += self.callbacks.on_epoch_end()
            if stop_signal:
                break

        return self.log

    def compute_g_case_control(self, case, control):
        '''We need to concat case and control and pass all though
        the net 'g' in one batch. If not, batch norm will fail.
        '''
        batch_size = case.size()[0]
        control = [ctr for ctr in control]
        both = torch.cat([case] + control)
        g_both = self.g(both)
        g_both = torch.split(g_both, batch_size)
        g_case = g_both[0]
        g_control = torch.stack(g_both[1:])
        return g_case, g_control

    @staticmethod
    def loss_func(g_case, g_control, clamp=(-3e+38, 88.)):
        '''Comput the loss = log[1 + sum(exp(g_control - g_case))]

        This is:
        loss = - g_case + log[exp(g_case) + sum(exp(g_control))]
        with a version of the log-sum-exp trick.

        Parameters:
        g_case: Torch array.
        g_control: List of arrays.
        clamp: Lower and upper cutoff for g_case and g_control. 88 is a nice max
            because exp(89) is inf for float32, which results in None gradients.
            One problem is that very large g_control will not have gradients (we
            want g_control to be small).
        '''
        control_sum = 0.
        for ctr in g_control:
            ctr = ctr - g_case
            ctr = torch.clamp(ctr, *clamp)  # Kills grads for very bad cases (should find better way).
            control_sum += torch.exp(ctr)
        loss = torch.log(1. + control_sum)
        return torch.mean(loss)
    
    def predict_g_numpy(self, X, batch_size=8224, return_numpy=True, eval_=True):
        '''Return g predictions, p = h0 exp(g(x)).

        Parameters:
            X: Numpy matrix with with covariates.
            batch_size: Batch size.
            return_numpy: If False, a torch tensor is returned.
            eval_: If true, set the network in eval mode for prediction
                and back to train mode after that (only affects dropout and batchnorm).
                If False, leaves the network modes as they are.
        '''
        return self._predict_func_numpy(self.g, X, batch_size, return_numpy, eval_)

    def predict_expg_numpy(self, X, batch_size=8224, return_numpy=True, eval_=True):
        '''Return g predictions, p = h0 exp(g(x)).

        Parameters:
            X: Numpy matrix with with covariates.
            batch_size: Batch size.
            return_numpy: If False, a torch tensor is returned.
            eval_: If true, set the network in eval mode for prediction
                and back to train mode after that (only affects dropout and batchnorm).
                If False, leaves the network modes as they are.
        '''
        expg = _Expg(self.g)
        return self._predict_func_numpy(expg, X, batch_size, return_numpy, eval_)

    def load_model_weights(self, path, warn=True, **kwargs):
        '''Load model weights.

        Parameters:
            path: The filepath of the model.
            **kwargs: Arguments passed to torch.load method.
        '''
        super().load_model_weights(path, **kwargs)
        if warn:
            warnings.warn('Might need to transfer to cuda???')
            warnings.warn('Need to recompute baseline hazards after loading.')
            warnings.warn('Might need to set optim again!')

    def _compute_baseline_hazards(self, df, max_duration, batch_size):
        '''Need to be implemeted by the respective methods'''
        raise NotImplementedError
    
    def compute_baseline_hazards(self, df=None, max_duration=None, sample=None, batch_size=8224,
                                set_hazards=True):
        '''Computes the breslow estimates of the baseline hazards of dataframe df.

        Parameters:
            df: Pandas dataframe with covariates, duration, and events.
                If None: use training data frame.
            max_duration: Don't compute hazards for durations larger than max_time.
            sample: Use sample of df. 
                Sample proportion if 'sample' < 1, else sample number 'sample'.
            batch_size: Batch size passed calculation of g_preds.
            set_hazards: If we should store computed hazards in object.

        Returns:
            Pandas series with baseline hazards. Index is duration_col.
        '''
        if df is None:
            if not hasattr(self, 'df'):
                raise ValueError('Need to fit a df, or supply a df to this function.')
            df = self.df
        
        if sample is not None:
            if sample >= 1:
                df = df.sample(n=sample)
            else:
                df = df.sample(frac=sample)
        
        base_haz = self._compute_baseline_hazards(df, max_duration, batch_size)

        if set_hazards:
            self.compute_baseline_cumulative_hazards(set_hazards=True, baseline_hazards_=base_haz)

        return base_haz

    def compute_baseline_cumulative_hazards(self, df=None, max_duration=None, sample=None,
                                            batch_size=8224, set_hazards=True, baseline_hazards_=None):
        '''Compute the baseline cumulative hazards of dataframe df or baseline_hazards.

        Parameters:
            df: Pandas dataframe with covariates, duration, and events.
                If None: use training data frame.
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
        if (df is not None) and (baseline_hazards_ is not None):
            raise ValueError('`df` and `baseline_hazards_` can not both be different from `None`.')
        if baseline_hazards_ is None:
            baseline_hazards_ = self.compute_baseline_hazards(df, max_duration, sample, batch_size,
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
    
    def predict_cumulative_hazards(self, df, max_duration=None, batch_size=16448, verbose=False, baseline_hazards_=None):
        '''Get cumulative hazards for dataset df.
        H(x, t) = sum [h0(t) exp(g(x, t))]
        or
        H(x, t) = sum [h0(t) exp(g(x))]

        Parameters:
            df: Pandas dataframe with covariates.
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
        return self._predict_cumulative_hazards(df, max_duration, batch_size, verbose, baseline_hazards_)

    def _predict_cumulative_hazards(self, df, max_duration, batch_size, verbose, baseline_hazards_):
        raise NotImplementedError

    def predict_survival_function(self, df, max_duration=None, batch_size=512, verbose=False, baseline_hazards_=None):
        '''Predict survival function for dataset df.
        S(x, t) = exp(-H(x, t))

        Parameters:
            df: Pandas dataframe with covariates.
            max_duration: Don't compute hazards for durations larger than max_time.
            batch_size: Batch size passed calculation of g_preds.
            verbose: If we should print progress.
            baseline_hazards_: Pandas series with index: time, and values: baseline hazards.

        Returns:
            Pandas data frame with survival functions. One columns for
            each individual in the df.
        '''
        return np.exp(-self.predict_cumulative_hazards(df, max_duration, batch_size, verbose, baseline_hazards_))

    def predict_cumulative_hazards_at_times(self, times, df, batch_size=16448, return_df=True,
                                           verbose=0, baseline_hazards_=None):
        '''Predict cumulative hazards only at given times. This is not as efficient as
        for the proportional hazards models.
        '''
        raise NotImplementedError

    def predict_survival_at_times(self, times, df, batch_size=16448, return_df=True,
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
        return np.exp(-self.predict_cumulative_hazards_at_times(times, df, batch_size, return_df,
                                                                verbose, baseline_hazards_))

    def brier_score(self, times, df, batch_size=512):
        '''Gives brier scores for `times`.

        Parameters:
            times: Number or iterable with times where to compute the score.
            df: Pandas dataframe with covariates, events, and durations.
            batch_size: Batch size passed calculation of g_preds.

        Returns:
            Numpy array with brier scores.
        '''
        prob_alive = self.predict_survival_at_times(times, df, batch_size, False)
        durations = df[self.duration_col].values
        events = df[self.event_col].values
        return brier_score(times, prob_alive, durations, events)

    def integrated_brier_score(self, df, times_grid=None, n_grid_points=100,
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
            return self.predict_survival_at_times(times, df, batch_size=batch_size, return_df=False)

        durations = df[self.duration_col].values
        events = df[self.event_col].values
        return integrated_brier_score(prob_alive_func, durations, events, times_grid, n_grid_points)


class CoxPH(CoxBase):
    '''This class implements fitting Cox's proportional hazards model:
    h(t|x) = h_0(t)*exp(g(x)), where g(x) is a neural net specified with pytorch.
    Parameters:
        g_model: pytorch net that implements the model g(x).
        optimizer: pytorch optimizer. If None optimizer is set to
            SGD with lr=0.01 and momentum=0.9.
        device: Which device to compute on.
            Preferrably pass a torch.device object.
            If `None`: use default gpu if avaiable, else use cpu.
            If `string`: string is passed to torch.device(`string`).
    '''
    @staticmethod
    def make_dataloader(X, durations, at_risk_dict, n_control, batch_size, num_workers):
        assert X.dtype == np.float32, 'Need Xtr to be np.float32'
        trainset = CoxPrepare(X, durations, at_risk_dict, n_control)
        dataloader = DataLoaderSlice(trainset, batch_size=batch_size, shuffle=True,
                                     num_workers=num_workers)
        return dataloader

    def predict_g(self, df, batch_size=8224, return_numpy=True, eval_=True):
        '''Return g(x) predictions, h = h0 * exp(g(x)).

        Parameters:
            df: Pandas dataframe with covariates.
            batch_size: Batch size.
            return_numpy: If False, a torch tensor is returned.
            eval_: If true, set the network in eval mode for prediction
                and back to train mode after that (only affects dropout and batchnorm).
                If False, leaves the network modes as they are.
        '''
        X = df[self.x_columns].as_matrix().astype('float32')
        return self.predict_g_numpy(X, batch_size, return_numpy, eval_)

    def predict_expg(self, df, batch_size=8224, return_numpy=True, eval_=True):
        '''Return exp(g(x)) predictions, h = h0 * exp(g(x)).

        Parameters:
            df: a Pandas dataframe with covariates.
            batch_size: batch size.
            return_numpy: If False, a torch tensor is returned
            eval_: If true, set the network in eval mode for prediction
                and back to train mode after that (only affects dropout and batchnorm).
                If False, leaves the network modes as they are.
        '''
        X = df[self.x_columns].as_matrix().astype('float32')
        return self.predict_expg_numpy(X, batch_size, return_numpy, eval_)

    def concordance_index(self, df, g_preds=None, batch_size=256):
        '''Concoradance index (from lifelines.utils).
        If g_preds are not supplied (None), they will be calculated.
            h = h0 * exp(g(x)).

        Parameters:
            df: Pandas dataframe with covariates, duration, and events.
            g_preds: Exponent of proportional hazards (h = h0 * exp(g(x))).
                If not supplied, it will be calculated.
            batch_size: Batch size passed calculation of g_preds.
        '''
        if g_preds is None:
            g_preds = self.predict_g(df, batch_size, True)
        return 1 - concordance_index(df[self.duration_col], g_preds.flatten(),
                                 df[self.event_col])

    def partial_log_likelihood(self, df, g_preds=None, batch_size=512):
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
        return (df
                .assign(_g_preds=g_preds if g_preds is not None else self.predict_g(df, batch_size, True))
                .sort_values(self.duration_col, ascending=False)
                .assign(_cum_exp_g=(lambda x: x['_g_preds']
                                    .pipe(np.exp)
                                    .cumsum()
                                    .groupby(x[self.duration_col])
                                    .transform('max')))
                .loc[lambda x: x[self.event_col] == 1]
                .assign(pll=lambda x: x['_g_preds'] - np.log(x['_cum_exp_g']))
                ['pll'])

    def _compute_baseline_hazards(self, df, max_duration, batch_size):
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
        return (df[[self.event_col, self.duration_col]]
                .assign(expg=self.predict_expg(df, batch_size))
                .groupby(self.duration_col)
                .agg({'expg': 'sum', self.event_col: 'sum'})
                .sort_index(ascending=False)
                .assign(expg=lambda x: x['expg'].cumsum())
                .pipe(lambda x: x[self.event_col]/x['expg'])
                .fillna(0.)
                .iloc[::-1]
                .loc[lambda x: x.index <= max_duration]
                .rename('baseline_hazards'))

    def _predict_cumulative_hazards(self, df, max_duration, batch_size, verbose, baseline_hazards_):
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
        expg = self.predict_expg(df, batch_size, return_numpy=True).reshape(1, -1)
        return pd.DataFrame(bch.values.reshape(-1, 1).dot(expg), columns=df.index,
                            index=bch.index)

    def predict_cumulative_hazards_at_times(self, times, df, batch_size=16448, return_df=True,
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
        expg = self.predict_expg(df, batch_size, return_numpy=True).reshape(1, -1)
        res = bch.dot(expg)
        if return_df:
            return pd.DataFrame(res, index=times, columns=df.index)
        return res


class CoxTime(CoxBase):
    '''Same as CoxNN, but we include time as a covariate.

    Possibly merge this class into CoxNN!!!!!!!

    time_as_cov: If time (duration) should be included a covariate.
    '''
    def _prepare_data_fit(self):
        '''Prepare data for fit.
        Add time_fail, gr_alive, x_columns, and Xtr to self.

         - Ensure corrent data types.
         - Get failure times: time_fail.
         - Get dict mapping from time to index of alive subjects: gr_alive.
         - Get feature matrix Xtr.
        '''
        if self.df[self.duration_col].dtype != 'float32':
            self.df = self.df.assign(**{self.duration_col: self.df[self.duration_col].astype('float32')})
        super()._prepare_data_fit()

    def fit(self, df, duration_col, event_col=None, n_control=1, batch_size=64, epochs=1,
            num_workers=0, verbose=1, strata=None, callbacks=None, compute_hazards=False):
        '''Fit the Cox Propertional Hazards model to a dataset, with time as a covariate.
        Tied survival times are handled using Efron's tie-method.

        Parameters:
            df: A Pandas dataframe with necessary columns `duration_col` and
                `event_col`, plus other covariates. `duration_col` refers to
                the lifetimes of the subjects. `event_col` refers to whether
                the 'death' events was observed: 1 if observed, 0 else (censored).
            duration_col: The column in dataframe that contains the subjects'
                lifetimes.
            event_col: The column in dataframe that contains the subjects' death
                observation. If left as None, assume all individuals are non-censored.
            n_control: Number of control samples.
            batch_size: Batch size.
            epochs: Number of epochs.
            num_workers: Number of workers for preparing data.
            verbose: Degree of verbose. If dict {'name': mm}, where mm is a MonitorMetric object,
                this will be printed. 
                Example: 
                mm = MonitorCoxTimeLoss(df_val, n_control=1, n_reps=4,)
                cox.fit(..., verbose={'val_loss': mm}, callbacks=[mm])
            strata: Specify a list of columns to use in stratification. This is useful if a
                catagorical covariate does not obey the proportional hazards assumption. This
                is used similar to the `strata` expression in R.
                See http://courses.washington.edu/b515/l17.pdf.
            callbacks: List of callbacks.
            compute_hazards: If we should compute hazards when training has finished.

        # Returns:
        #     self, (with additional properties: hazards_)
        '''
        if compute_hazards:
            warnings.warn('''
            Computing hazards over the full trainingset might be very expensive.
            Consider setting `compute_hazards`=False, and use 
            model.compute_baseline_hazards() instead.
            ''')
        return super().fit(df, duration_col, event_col, n_control, batch_size, epochs,
                           num_workers, verbose, strata, callbacks, compute_hazards)

    @staticmethod
    def make_dataloader(X, durations, at_risk_dict, n_control, batch_size, num_workers):
        assert X.dtype == np.float32, 'Need Xtr to be np.float32'
        assert durations.dtype == 'float32', 'To use time as a covariate, we need dtype: float32'
        trainset = CoxPrepareWithTime(X, durations, at_risk_dict, n_control)
        dataloader = DataLoaderSlice(trainset, batch_size=batch_size, shuffle=True,
                                     num_workers=num_workers)
        return dataloader

    def predict_g(self, df, batch_size=512, return_numpy=True, eval_=True):
        '''Return g(x, t) predictions, h = h0 * exp(g(x, t)).

        Parameters:
            df: Pandas dataframe with covariates and time.
            batch_size: Batch size.
            return_numpy: If False, a torch tensor is returned.
            eval_: If true, set the network in eval mode for prediction
                and back to train mode after that (only affects dropout and batchnorm).
                If False, leaves the network modes as they are.
        '''
        cols = list(self.x_columns) + [self.duration_col]
        x = df[cols].as_matrix().astype('float32')
        return self.predict_g_numpy(x, batch_size, return_numpy, eval_)

    def predict_expg(self, df, batch_size=512, return_numpy=True, eval_=True):
        '''Return g(x, t) predictions, h = h0 * exp(g(x, t)).

        Parameters:
            df: Pandas dataframe with covariates and time.
            batch_size: Batch size.
            return_numpy: If False, a torch tensor is returned.
            eval_: If true, set the network in eval mode for prediction
                and back to train mode after that (only affects dropout and batchnorm).
                If False, leaves the network modes as they are.
        '''
        cols = list(self.x_columns) + [self.duration_col]
        x = df[cols].as_matrix().astype('float32')
        return self.predict_expg_numpy(x, batch_size, return_numpy, eval_)

    def _compute_baseline_hazards(self, df, max_duration, batch_size):
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
            expg = self.predict_expg(df.iloc[ix:].assign(**{self.duration_col: t}), batch_size)
            return expg.flatten().sum()

        df = df.sort_values(self.duration_col).reset_index(drop=True)
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

    def _predict_cumulative_hazards(self, df, max_duration, batch_size, verbose, baseline_hazards_):
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
            return self.predict_expg(df.assign(**{self.duration_col: t}), batch_size).flatten()

        max_duration = np.inf if max_duration is None else max_duration
        baseline_hazards_ = baseline_hazards_.loc[lambda x: x.index <= max_duration]
        r, c = baseline_hazards_.shape[0], df.shape[0]
        hazards = np.empty((r, c))
        for idx, t in enumerate(baseline_hazards_.index):
            if verbose:
                print(idx, 'of', len(baseline_hazards_))
            hazards[idx, :] = expg_at_time(t)
        hazards *= baseline_hazards_.values.reshape(-1, 1)
        return pd.DataFrame(hazards, columns=df.index, index=baseline_hazards_.index).cumsum()

    def predict_cumulative_hazards_at_times(self, times, df, batch_size=16448, return_df=True,
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
        cum_haz = self.predict_cumulative_hazards(df, max_duration, batch_size,
                                                  verbose, baseline_hazards_)
        times_idx = search_sorted_idx(cum_haz.index.values, times)
        cum_haz = cum_haz.iloc[times_idx]
        if return_df:
            return cum_haz
        return cum_haz.as_matrix()

    def partial_log_likelihood(self, df, batch_size=512):
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
            return self.predict_expg(df.iloc[i:].assign(**{self.duration_col: t}), batch_size).flatten().sum()

        df = df.sort_values(self.duration_col)

        times =  (df[[self.duration_col, self.event_col]]
                  .assign(_idx=np.arange(len(df)))
                  .loc[lambda x: x[self.event_col] == True]
                  .drop_duplicates(self.duration_col, keep='first')
                  .assign(_expg_sum=lambda x: [expg_sum(t, i) for t, i in zip(x[self.duration_col], x['_idx'])])
                  .drop([self.event_col, '_idx'], axis=1))
        
        idx_name_old = df.index.name
        idx_name = '__' + idx_name_old if idx_name_old else '__index'
        df.index.name = idx_name

        pll = (df
               .loc[lambda x: x[self.event_col] == True]
               .assign(_g_preds=lambda x: self.predict_g(x, batch_size=batch_size).flatten())
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


class CoxPHFunc(CoxPH):
    '''Class for doing same as CoxNN on an arbitrary funciton g.
    h(x, t) = h_0(t) * exp(g(x))

    Parameters:
        func: Function e.g. lambda x: x**2.
        df: Training pandas dataframe.
        duration_col: Name of column in df giving durations.
        event_col: Name of column in df giving events.
    '''
    def __init__(self, func, df, duration_col, event_col):
        optimizer = 'Not defined in this class'
        g_model = FuncTorch(func)
        super().__init__(g_model, optimizer)
        self._fake_fit(df, duration_col, event_col)

    def predict_g(self, df, *args, **kwargs):
        x = df[self.x_columns].as_matrix().astype('float32')
        return self.g(x)
    
    def predict_expg(self, df, *args, **kwargs):
        return np.exp(self.predict_g(df, *args, **kwargs))

    def _fake_fit(self, df, duration_col, event_col):
        return super().fit(df, duration_col, event_col, epochs=0)

    def fit(self, *args, **kwargs):
        '''It's not possible to fit this object.'''
        raise ValueError("It's not possible to fit this object")

class CoxTimeFunc(CoxTime):
    '''Class for doing same as CoxTime on an arbitrary funciton g.

    Parameters:
        func: Function that g(x):
        df: Training pandas dataframe.
        duration_col: Name of column in df giving durations.
        event_col: Name of column in df giving events.
    '''
    def __init__(self, g_func, df, duration_col, event_col):
        optimizer = 'Not defined in this class'
        g_model = FuncTorch(g_func)
        super().__init__(g_model, optimizer)
        self._fake_fit(df, duration_col, event_col)

    def predict_g(self, df, *args, **kwargs):
        cols = list(self.x_columns) + [self.duration_col]
        x = df[cols].as_matrix().astype('float32')
        return self.g(x)
    
    def predict_expg(self, df, *args, **kwargs):
        return np.exp(self.predict_g(df, *args, **kwargs))

    def _fake_fit(self, df, duration_col, event_col):
        return super().fit(df, duration_col, event_col, epochs=0)

    def fit(self, *args, **kwargs):
        '''It's not possible to fit this object.'''
        raise ValueError("It's not possible to fit this object")

class CoxLifelines(CoxPHFunc):
    '''Class for doing same as CoxPH on lifelines cph object.

    Parameters:
        cph_lifelines: Fitted CoxPHFitter object from Lifelines.
        df: Training pandas dataframe.
        duration_col: Name of column in df giving durations.
        event_col: Name of column in df giving events.
    '''
    def __init__(self, cph_lifelines, df, duration_col, event_col):
        self.cph_lifelines = cph_lifelines
        func = lambda x: np.log(self.cph_lifelines.predict_partial_hazard(x)).as_matrix()
        super().__init__(func, df, duration_col, event_col)
