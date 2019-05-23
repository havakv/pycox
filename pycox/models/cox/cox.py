import os
import warnings
import numpy as np
import pandas as pd
from torchtuples import Model, tuplefy, make_dataloader
from pycox.models.cox.data import DatasetDurationSorted

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


class CoxBase(Model):
    duration_col = 'duration'
    event_col = 'event'

    def fit(self, input, target, batch_size=256, epochs=1, callbacks=None, verbose=True,
            num_workers=0, shuffle=True, metrics=None, val_data=None, val_batch_size=8224,
            **kwargs):
        """Fit  model with inputs and targets. Where 'input' is the covariates, and
        'target' is a tuple with (durations, events).
        
        Arguments:
            input {np.array, tensor or tuple} -- Input x passed to net.
            target {np.array, tensor or tuple} -- Target [durations, events]. 
        
        Keyword Arguments:
            batch_size {int} -- Elemets in each batch (default: {256})
            epochs {int} -- Number of epochs (default: {1})
            callbacks {list} -- list of callbacks (default: {None})
            verbose {bool} -- Print progress (default: {True})
            num_workers {int} -- Number of workers used in the dataloader (default: {0})
            shuffle {bool} -- If we should shuffle the order of the dataset (default: {True})
            **kwargs are passed to 'make_dataloader' method.
    
        Returns:
            TrainingLogger -- Training log
        """
        self.training_data = tuplefy(input, target)
        return super().fit(input, target, batch_size, epochs, callbacks, verbose,
                           num_workers, shuffle, metrics, val_data, val_batch_size,
                           **kwargs)

    def _compute_baseline_hazards(self, input, df, max_duration, batch_size):
        raise NotImplementedError

    def target_to_df(self, target):
        durations, events = tuplefy(target).to_numpy()
        df = pd.DataFrame({self.duration_col: durations, self.event_col: events}) 
        return df

    def compute_baseline_hazards(self, input=None, target=None, max_duration=None, sample=None, batch_size=8224,
                                set_hazards=True):
        """Computes the Breslow estimates form the data definded by `input` and `target`
        (if `None` use traning data).

        Typically call
        model.compute_baseline_hazards() after fitting.
        
        Keyword Arguments:
            input  -- Input data (train input) (default: {None})
            target  -- Taget data (train target) (default: {None})
            max_duration {float} -- Don't compute estimates for duration higher (default: {None})
            sample {float or int} -- Compute estimates of subsample of data (default: {None})
            batch_size {int} -- Batch size (default: {8224})
            set_hazards {bool} -- Set hazards in model object, or just return hazards. (default: {True})
        
        Returns:
            pd.Series -- Pandas series with baseline hazards. Index is duration_col.
        """
        if (input is None) and (target is None):
            if not hasattr(self, 'training_data'):
                raise ValueError("Need to give a 'input' and 'target' to this function.")
            input, target = self.training_data
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
        """See `compute_bseline_hazards. This is the cumulative version."""
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
        """See `predict_survival_function`."""
        if type(input) is pd.DataFrame:
            input = self.df_to_input(input)
        if baseline_hazards_ is None:
            if not hasattr(self, 'baseline_hazards_'):
                raise ValueError('Need to compute baseline_hazards_. E.g run `model.compute_baseline_hazards()`')
            baseline_hazards_ = self.baseline_hazards_
        assert baseline_hazards_.index.is_monotonic_increasing,\
            'Need index of baseline_hazards_ to be monotonic increasing, as it represents time.'
        return self._predict_cumulative_hazards(input, max_duration, batch_size, verbose, baseline_hazards_)

    def _predict_cumulative_hazards(self, input, max_duration, batch_size, verbose, baseline_hazards_):
        raise NotImplementedError

    def predict_survival_function(self, input, max_duration=None, batch_size=8224, verbose=False, baseline_hazards_=None):
        """Predict survival function for `input`. S(x, t) = exp(-H(x, t))
        Require compueted baseline hazards.

        Arguments:
            input {np.array, tensor or tuple} -- Input x passed to net.

        Keyword Arguments:
            max_duration {float} -- Don't compute estimates for duration higher (default: {None})
            batch_size {int} -- Batch size (default: {8224})
            baseline_hazards_ {pd.Series} -- Baseline hazards. If `None` used `model.baseline_hazards_` (default: {None})

        Returns:
            pd.DataFrame -- Survival esimates. One columns for each individual.
        """
        return np.exp(-self.predict_cumulative_hazards(input, max_duration, batch_size, verbose, baseline_hazards_))

    def predict_cumulative_hazards_at_times(self, times, input, batch_size=8224, return_df=True,
                                            verbose=False, baseline_hazards_=None):
        """NOTE: Don't know if this still works!!!!

        See `predict_cumulative_hazards`
        """
        raise NotImplementedError

    def predict_survival_at_times(self, times, input, batch_size=8224, return_df=True,
                                  verbose=False, baseline_hazards_=None):
        """NOTE: Don't know if this still works!!!!
        
        Predict survival function for `input` at five time points. S(x, t) = exp(-H(x, t))
        Require compueted baseline hazards.

        See `predict_survival_function`
        """
        return np.exp(-self.predict_cumulative_hazards_at_times(times, input, batch_size, return_df,
                                                                verbose, baseline_hazards_))

    def save_net(self, path, **kwargs):
        """Save self.net and baseline hazards to file.

        Arguments:
            path {str} -- Path to file.
            **kwargs are passed to torch.save

        Returns:
            None
        """
        path_list = path.split('.')
        extension = 'pt'
        if len(path_list) > 1:
            path = path_list[0]
            extension = path_list[1]
        extension = '.'+extension
        super().save_net(path+extension, **kwargs)
        if hasattr(self, 'baseline_hazards_'):
            self.baseline_hazards_.to_pickle(path+'_blh.pickle')

    def load_net(self, path, **kwargs):
        """Load net and hazards from file.

        Arguments:
            path {str} -- Path to file.
            **kwargs are passed to torch.load

        Returns:
            None
        """
        path_list = path.split('.')
        extension = 'pt'
        if len(path_list) > 1:
            path = path_list[0]
            extension = path_list[1]
        extension = '.'+extension
        super().load_net(path+extension, **kwargs)
        blh_path = path+'_blh.pickle'
        if os.path.isfile(blh_path):
            self.baseline_hazards_ = pd.read_pickle(blh_path)
            self.baseline_cumulative_hazards_ = self.baseline_hazards_.cumsum()

    def df_to_input(self, df):
        input = df[self.input_cols].values
        return input
    
    def df_to_target(self, df):
        target = (df[self.duration_col].values, df[self.event_col].values)
        return tuplefy(target)

    def fit_df(self, df, duration_col, event_col, batch_size=256, epochs=1, callbacks=None,
               verbose=True, num_workers=0, shuffle=True, metrics=None, val_df=None,
               val_batch_size=8224, n_control=1, **kwargs):
        """NOTE: Don't know if this still works. Use `fit` instead.

        Fit the Cox Propertional Hazards model to a dataset. Tied survival times
        are handled using Beslow's tie-method.

        Parameters:
            df: A Pandas dataframe with necessary columns `duration_col` and
                `event_col`, plus other covariates. `duration_col` refers to
                the lifetimes of the subjects. `event_col` refers to whether
                the 'death' events was observed: 1 if observed, 0 else (censored).
            duration_col: The column in dataframe that contains the subjects'
                lifetimes.
            event_col: The column in dataframe that contains the subjects' death
                observation. 
            batch_size: Batch size.
            epochs: Number of epochs.
            num_workers: Number of workers for preparing data.
            verbose: Degree of verbose. If dict {'name': mm}, where mm is a MonitorMetric object,
                this will be printed. 
                Example: 
                mm = MonitorCoxLoss(df_val, n_control=1, n_reps=4,)
                cox.fit(..., verbose={'val_loss': mm}, callbacks=[mm])
            callbacks: List of callbacks.
            n_control: Number of control samples.
            compute_hazards: If we should compute hazards when training has finished.

        # Returns:
        #     self, with additional properties: hazards_
        """
        self.duration_col = duration_col
        self.event_col = event_col
        self.input_cols = df.columns.drop([self.duration_col, self.event_col]).values
        input, target = self.df_to_input(df), self.df_to_target(df)
        val_data = val_df
        if val_data is not None:
            val_data = self.df_to_input(val_data), self.df_to_target(val_data)
        return self.fit(input, target, batch_size, epochs, callbacks, verbose, num_workers,
                           shuffle, metrics, val_data, val_batch_size, n_control=n_control, **kwargs)

    def compute_baseline_hazards_df(self, df=None, max_duration=None, sample=None, batch_size=8224,
                                set_hazards=True):
        """See `compute_baeline_hazards`"""
        input, target = None, None
        if df is not None:
            input, target = self.df_to_input(df), self.df_to_target
        return self.compute_baseline_hazards_df(input, target, max_duration, sample, batch_size,
                                set_hazards)

    def compute_baseline_cumulative_hazards_df(self, df=None, max_duration=None, sample=None,
                                            batch_size=8224, set_hazards=True, baseline_hazards_=None):
        """See `compute_baseline_cumulative_hazards`."""
        input, target = None, None
        if df is not None:
            input, target = self.df_to_input(df), self.df_to_target
        return self.compute_baseline_cumulative_hazards(input, target, max_duration, sample,
                                                        batch_size, set_hazards, baseline_hazards_)

    def partial_log_likelihood_df(self, df, g_preds=None, batch_size=8224):
        '''See `partial_log_likelihood`.

        Calculate the partial log-likelihood for the events in datafram df.
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
        input, target = self.df_to_input(df), self.df_to_target
        return self.partial_log_likelihood(input, target, g_preds, batch_size)


class CoxPHBase(CoxBase):
    def _compute_baseline_hazards(self, input, df_target, max_duration, batch_size):
        if max_duration is None:
            max_duration = np.inf

        # Here we are computing when expg when there are no events.
        #   Could be made faster, by only computing when there are events.
        return (df_target
                .assign(expg=np.exp(self.predict(input, batch_size, numpy=True)))
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
        max_duration = np.inf if max_duration is None else max_duration
        if baseline_hazards_ is self.baseline_hazards_:
            bch = self.baseline_cumulative_hazards_
        else:
            bch = self.compute_baseline_cumulative_hazards(set_hazards=False, 
                                                           baseline_hazards_=baseline_hazards_)
        bch = bch.loc[lambda x: x.index <= max_duration]
        expg = np.exp(self.predict(input, batch_size, numpy=True)).reshape(1, -1)
        return pd.DataFrame(bch.values.reshape(-1, 1).dot(expg), 
                            index=bch.index)

    def predict_cumulative_hazards_at_times(self, times, input, batch_size=8224, return_df=True,
                                            verbose=False, baseline_hazards_=None):
        if type(input) is pd.DataFrame:
            input = self.df_to_input(input)
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
        expg = np.exp(self.predict(input, batch_size, numpy=True)).reshape(1, -1)
        res = bch.dot(expg)
        if return_df:
            return pd.DataFrame(res, index=times)
        return res

    def partial_log_likelihood(self, input, target, g_preds=None, batch_size=8224, eps=1e-7):
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
            g_preds = self.predict(input, batch_size, numpy=True)
        return (df
                .assign(_g_preds=g_preds)
                .sort_values(self.duration_col, ascending=False)
                .assign(_cum_exp_g=(lambda x: x['_g_preds']
                                    .pipe(np.exp)
                                    .cumsum()
                                    .groupby(x[self.duration_col])
                                    .transform('max')))
                .loc[lambda x: x[self.event_col] == 1]
                .assign(pll=lambda x: x['_g_preds'] - np.log(x['_cum_exp_g'] + eps))
                ['pll'])


class CoxPH(CoxPHBase):
    """Cox proportional hazards model parameterized with a neural net.
    This is essentailly the DeepSurv method by Katzman et al. (2018)
    https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/s12874-018-0482-1

    The loss function is not quite the parial log-likelihood, but close.    
    The difference is that for tied events, we use a random order instead of 
    including all individuals that had an event at that point in time.

    Arguments:
        net {torch.nn.Module} -- A pytorch net.
    
    Keyword Arguments:
        optimizer {torch or torchtuples optimizer} -- Optimizer (default: {None})
        device {string, int, or torch.device} -- See torchtuples.Model (default: {None})
    """
    def __init__(self, net, optimizer=None, device=None):
        loss = loss_cox_ph
        return super().__init__(net, loss=loss, optimizer=optimizer, device=device)

    @staticmethod
    def make_dataloader(data, batch_size, shuffle, num_workers=0):
        dataloader = make_dataloader(data, batch_size, shuffle, num_workers,
                                     make_dataset=DatasetDurationSorted)
        return dataloader

    def make_dataloader_predict(self, input, batch_size, shuffle=False, num_workers=0):
        dataloader = super().make_dataloader(input, batch_size, shuffle, num_workers)
        return dataloader


def loss_cox_ph(log_h, event, eps=1e-7):
    """Requires the input to be sorted by descending duration time.
    See DatasetDurationSorted.

    We calculate the negative log of $(\frac{h_i}{\sum_{j \in R_i} h_j})^d$,
    where h = exp(log_h) are the hazards and R is the risk set, and d is event.

    We just compute a cumulative sum, and not the true Risk sets. This is a
    limitiation, but simple and fast.
    """
    event = event.view(-1)
    log_h = log_h.view(-1)
    gamma = log_h.max()
    log_cumsum_h = log_h.sub(gamma).exp().cumsum(0).add(eps).log().add(gamma)
    return - log_h.sub(log_cumsum_h).mul(event).sum().div(event.sum())

