'''
File contains an implementation of cox regression with arbirary neural network as input.
'''
import warnings
import numpy as np
# import scipy
import pandas as pd

import torch
from torch.autograd import Variable
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
# import torch.utils.data as data

# from kds.pytorch_help import DataLoaderBatch

# from sklearn.metrics import roc_auc_score
# from lifelines.utils import concordance_index
# from lifelines import KaplanMeierFitter

from .callbacks import CallbacksList, TrainingLogger, EarlyStoppingTrainLoss
from .dataloader import DataLoaderBatch, CoxPrepare, CoxPrepareWithTime, NumpyTensorDataset
from .metrics import concordance_index, brier_score, integrated_brier_score




class CoxNNT(object):
    '''Class holding the cox method for temportal covariates.
    See CoxNN for static covariates.
    TODO:
     - Use high number of controls (more than whats alwas available):
        - Combine batches with similar controls (can have mixing problems).
     ez - batch_size=1 can work quite easily (but slow).
     EZ - Use copies of same control when there is not enough (e.g. sample with replacement)
        - Use copies, but add something canceling out the gradients.

    gModel: pytorch model for computing g(X). h = h0 exp(g(X)).
    optimizer: torch optimizer. If None SGD(lr=0.1, momentum=0.9)
    cuda: if we should run on a gpu if available.
    '''
    def __init__(self, gModel, optimizer=None, cuda=True):
        self.g = gModel
        self.optimizer = optimizer
        if optimizer is None:
            self.optimizer = optim.SGD(self.g.parameters(), lr=0.01, momentum=0.9)
        self.log = TrainingLogger()
        self.cuda = False
        if cuda and torch.cuda.is_available():
            self.cuda = True
            self.g.cuda()

    def fit(self, Xtr, time_fail, gr_alive, n_control=1,
            batch_size=64, epochs=1, n_workers=0, verbose=1, callbacks=[]):
        '''Use fit_dataloader of you prefer to give a DataLoader instead.

        Parametes:
            Xtr: Numpy array with covariates.
            time_fail: Pandas series with index same as fails in Xtr, and values
                giving time of that failure.
            gr_alive: Dict with gr_alive[time] = <array with index of alive in X matrix>.
            n_control: Number of control samples.
            batch_size: Batch size.
            epochs: Number of epochs.
            n_workers: Number of workers for preparing data.
            verbose: Degree of verbose.
            callbacks: List of callbacks.
        '''
        dataloader = self._make_dataloader(Xtr, time_fail, gr_alive, n_control, batch_size,
                                           n_workers)
        log = self.fit_dataloader(dataloader, epochs, verbose, callbacks)
        return log

    @staticmethod
    def _make_dataloader(Xtr, time_fail, gr_alive, n_control, batch_size, n_workers):
        assert Xtr.dtype == np.float32, 'Need Xtr to be np.float32'
        trainset = CoxPrepare(Xtr, time_fail, gr_alive, n_control)
        dataloader = DataLoaderBatch(trainset, batch_size=batch_size, shuffle=True,
                                     num_workers=n_workers)
        return dataloader

    def fit_dataloader(self, dataloader, epochs=1, verbose=1, callbacks=None):
        '''Fit method for pytorch dataloader.'''
        self.fit_info = {'batches_per_epoch': len(dataloader)}

        self.log.verbose = verbose
        if callbacks is None:
            callbacks = []
        self.callbacks = CallbacksList([self.log]+callbacks)
        self.callbacks.give_model(self)

        self.callbacks.on_fit_start()
        for _ in range(epochs):
            for case, control in dataloader:
                if self.cuda:
                    case, control = case.cuda(), control.cuda()
                case, control = Variable(case), Variable(control)
                self.fit_info['case_control'] = (case, control)
                gCase = self.g(case)
                gControl = [self.g(ctr) for ctr in control]
                self.fit_info['g_case_control'] = (gCase, gControl)
                self.batch_loss = self.compute_loss(gCase, gControl)
                self.optimizer.zero_grad()
                self.batch_loss.backward()
                stop_signal = self.callbacks.before_step()
                if stop_signal:
                    raise RuntimeError('Stop signal in before_step().')
                self.optimizer.step()
                self.callbacks.on_batch_end()
            stop_signal = self.callbacks.on_epoch_end()
            if stop_signal:
                break

        return self.log

    @staticmethod
    def compute_loss(g_case, g_control, log_eps=1e-38, clamp=(-3e+38, 88)):
        '''Comput the loss = - g_case + log[exp(g_case) + sum(exp(g_control))]

        log_eps: 1e-38 is around the smalles value we can use without getting
            nan gradients. This is found by experimenting...
        clamp: Lower and upper cutoff for g_case and g_control. 88 is a nice max
            because exp(89) is inf for float32, which results in None gradients.
            One problem is that very large g_control will not have gradients (we
            want g_control to be small).
        
        TODO:
            Use log-sum-exp trick for better numerical stability.
                log(sum(exp(x))) = a + log(sum(exp(x-a))),
                where a is calculated from the batch???
        '''
        control_sum = 0.
        for ctr in g_control:
            ctr = torch.clamp(ctr, *clamp)  # Kills grads for very bad cases (should find better way).
            control_sum += torch.exp(ctr)
        g_case = torch.clamp(g_case, *clamp)  # This is somewhat fine as higher than 88 is unlikely in real world.
        loss = - g_case + torch.log(torch.exp(g_case) + control_sum + log_eps)
        return torch.mean(loss)

    def predict_g(self, X, batch_size=512, return_numpy=True, eval_=True):
        '''Return g predictions, p = h0 exp(g(x)).

        Parameters:
            X: Numpy matrix with with covariates.
            batch_size: Batch size.
            return_numpy: If False, a torch tensor is returned.
            eval_: If true, set the network in eval mode for prediction
                and back to train mode after that (only affects dropout and batchnorm).
                If False, leaves the network modes as they are.
        '''
        if eval_:
            self.g.eval()
        if len(X) < batch_size:
            if self.cuda:
                preds = [self.g(Variable(torch.from_numpy(X).cuda(), volatile=True))]
            else:
                preds = [self.g(Variable(torch.from_numpy(X), volatile=True))]
        else:
            dataset = NumpyTensorDataset(X)
            dataloader = DataLoaderBatch(dataset, batch_size)
            if self.cuda:
                preds = [self.g(Variable(x.cuda(), volatile=True))
                         for x in iter(dataloader)]
            else:
                preds = [self.g(Variable(x, volatile=True))
                         for x in iter(dataloader)]
        if eval_:
            self.g.train()
        if return_numpy:
            if self.cuda:
                preds = [pred.data.cpu().numpy() for pred in preds]
            else:
                preds = [pred.data.numpy() for pred in preds]
            return np.concatenate(preds)
        return preds

    def save_model_weights(self, path, **kwargs):
        '''Save the model weights.

        Parameters:
            path: The filepath of the model.
            **kwargs: Arguments passed to torch.save method.
        '''
        torch.save(self.g.state_dict(), path, **kwargs)

    def load_model_weights(self, path, warn=True, **kwargs):
        '''Load model weights.

        Parameters:
            path: The filepath of the model.
            **kwargs: Arguments passed to torch.load method.
        '''
        self.g.load_state_dict(torch.load(path, **kwargs))
        if warn:
            warnings.warn('Need to recompute baseline hazards after loading.')
            warnings.warn('Might need to set optim again!')


class CoxPH(CoxNNT):
    '''This class implements fitting Cox's proportional hazard model:
    h(t|x) = h_0(t)*exp(g(x)), where g(x) is a neural net specified with pytorch.
    Parameters:
        gModel: pytorch net that implements the model g(x).
        optimizer: pytorch optimizer. If None optimizer is set to
            SGD with lr=0.01 and momentum=0.9.
        cuda: Set to True if use GPU.
    '''
    def __init__(self, g_model, optimizer=None, cuda=False):
        self.g = g_model
        self.optimizer = optimizer
        if optimizer is None:
            self.optimizer = optim.SGD(self.g.parameters(), lr=0.01, momentum=0.9)
        self.cuda = False
        if cuda:
            if not torch.cuda.is_available():
                raise ValueError('Cude is not available')
            self.cuda = True
            self.g.cuda()
        self.log = TrainingLogger()

    def _is_repeated_fit_df(self, df, duration_col, event_col):
        if hasattr(self, 'df'):
            if (self._df is df) and (self.duration_col == duration_col) and (self.event_col == event_col):
                return True
        return False

    def _prepare_data_fit(self):
        '''Prepare data for fit.
        Add time_fail, gr_alive, x_columns, and Xtr to self.

         - Ensure corrent data types.
         - Get failure times: time_fail.
         - Get dict mapping from time to index of alive subjects: gr_alive.
         - Get feature matrix Xtr.
        '''
        self.time_fail = self.df.loc[lambda x: x[self.event_col] == 1][self.duration_col]
        self.gr_alive = self._gr_alive(self.df, self.duration_col)
        Xtr = self.df.drop([self.duration_col, self.event_col, self.old_index_name], axis=1)
        self.x_columns = Xtr.columns
        self.Xtr = Xtr.as_matrix().astype('float32')

    def fit(self, df, duration_col, event_col=None, n_control=1, batch_size=64, epochs=1,
            n_workers=0, verbose=1, strata=None, callbacks=None, compute_hazards=True):
        '''Fit the Cox Propertional Hazard model to a dataset. Tied survival times
        are handled using Efron's tie-method.

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
            n_workers: Number of workers for preparing data.
            strata: Specify a list of columns to use in stratification. This is useful if a
                catagorical covariate does not obey the proportional hazard assumption. This
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
        log = super().fit(self.Xtr, self.time_fail, self.gr_alive, n_control,
                          batch_size, epochs, n_workers, verbose, callbacks)

        self.set_hazards(self.df, compute_hazards=compute_hazards)
        return log

    def set_hazards(self, df=None, batch_size=512, compute_hazards=True):
        '''Add attributes baseline_hazard_ and baseline_cumulative_hazard_ to self.

        Parameters:
            df: Pandas dataframe with covariates, duration, and events.
                If None: use training data frame.
            batch_size: Batch size passed calculation of g_preds.
            compute_hazards: If False, set hazards to None.

        Returns:
            Nothing
        '''
        if compute_hazards: 
            self.baseline_hazard_ = self.compute_baseline_hazard(df, batch_size=batch_size)
            self.baseline_cumulative_hazard_ =\
                self.compute_baseline_cumulative_hazard(baseline_hazard=self.baseline_hazard_,
                                                        batch_size=batch_size)
        else:
            self.baseline_hazard_, self.baseline_cumulative_hazard_ = None, None

    @staticmethod
    def _reset_idx_with_old_idx_name(df):
        '''Reset index of data frame, and return df with
        index column name.
        '''
        cols = set(df.columns)
        df = df.reset_index()
        col = list(set(df.columns) - cols)[0]
        return df, col

    @staticmethod
    def _gr_alive(df, duration_col):
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

        gr_alive = dict()
        for ix, t in keys.iteritems():
            gr_alive[t] = allidx[ix:]
        return gr_alive

    def predict_g(self, df, batch_size=512, return_numpy=True, eval_=True):
        '''Return g(x) predictions, h = h0 * exp(g(x)).

        Parameters:
            df: Pandas dataframe with covariates.
            batch_size: Batch size.
            return_numpy: If False, a torch tensor is returned.
            eval_: If true, set the network in eval mode for prediction
                and back to train mode after that (only affects dropout and batchnorm).
                If False, leaves the network modes as they are.
        '''
        x = df[self.x_columns].as_matrix().astype('float32')
        return super().predict_g(x, batch_size, return_numpy, eval_)

    def predict_expg(self, df, batch_size=512, return_numpy=True):
        '''Return exp(g(x)) predictions, h = h0 * exp(g(x)).

        Parameters:
            df: a Pandas dataframe with covariates.
            batch_size: batch size.
            return_numpy: If False, a torch tensor is returned
        '''
        if return_numpy is False:
            raise NotImplementedError('Only implemented for numpy now.')
        return np.exp(self.predict_g(df, batch_size, return_numpy=True))

    def predict(self, df, batch_size=64, compute_on_log=True):
        '''Predict hazard h = h0 * exp(g(X))
        Requires that we have computed the breslow estimates for h0.
        X: covariates.
        times: DataFrame with same index as X, the times
            corresponding to the covariates (timeAlive), and the
            label (failNextDay).
        compute_on_log: If True, we compute exp(g(x) + log(h0)), while
            if False, we compute exp(g(x)) * h0. They should be equivalent,
            but True is recomended due to numerical stability.
        '''
        raise NotImplementedError()

    def concordance_index(self, df, g_preds=None, batch_size=256):
        '''Concoradance index (from lifelines.utils).
        If g_preds are not supplied (None), they will be calculated.
            h = h0 * exp(g(x)).

        Parameters:
            df: Pandas dataframe with covariates, duration, and events.
            g_preds: Exponent of proportional hazard (h = h0 * exp(g(x))).
                If not supplied, it will be calculated.
            batch_size: Batch size passed calculation of g_preds.
        '''
        if g_preds is None:
            g_preds = self.predict_g(df, batch_size, True)
        return concordance_index(df[self.duration_col], g_preds.flatten(),
                                 df[self.event_col])

    def partial_log_likelihood(self, df, g_preds=None, batch_size=512):
        '''Calculate the partial log-likelihood for the events in datafram df.
        This likelihood does not sample the controls.
        Note that censored data (non events) does not have a partial log-likelihood.

        Parameters:
            df: Pandas dataframe with covariates, duration, and events.
            g_preds: Exponent of proportional hazard (h = h0 * exp(g(x))).
                If not supplied, it will be calculated.
            batch_size: Batch size passed calculation of g_preds.

        Returns:
            Pandas dataframe with duration, g_preds, and the
                partial log-likelihood pll.
        '''
        if df is self._df:  # We use the trainig df
            warnings.warn('Should make this more effective when we use training df')
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

    def compute_baseline_hazard(self, df=None, batch_size=512):
        '''Computes the breslow estimates of the baseline hazards of dataframe df.

        Parameters:
            df: Pandas dataframe with covariates, duration, and events.
                If None: use training data frame.
            batch_size: Batch size passed calculation of g_preds.

        Returns:
            Pandas series with baseline hazard. Index is duration_col.
        '''
        if df is None:
            if not hasattr(self, 'df'):
                raise ValueError('Need to fit a df, or supply a df to this function.')
            df = self.df
            # warnings.warn("For train df, we should not recompute the groupby")

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
                .rename('baseline_hazard'))

    def compute_baseline_cumulative_hazard(self, df=None, baseline_hazard=None, batch_size=512):
        '''Compute the baseline cumulative hazard of dataframe df or baseline_hazard.

        Parameters:
            df: Pandas dataframe with covariates, duration, and events.
                If `None` use training data frame, or supplied baseline_hazards.
            baseline_hazards: Pandas series with baseline hazards.
                If `None` use supplied df or training data frame.
            batch_size: Batch size passed calculation of g_preds.

        Returns:
            Pandas series with baseline cumulative hazard. Index is duration_col.
        '''
        if baseline_hazard is None:
            if df is None:
                raise ValueError('`df` and `baseline_hazards`\
                                 cannot both be `None`.')
            baseline_hazard = self.compute_baseline_hazard(df, batch_size=batch_size)
        else:
            if df is not None:
                raise ValueError('Only one of `df` and `baseline_hazards`\
                                 can be different from None.')
        return (baseline_hazard
                .cumsum()
                .rename('baseline_cumulative_hazard'))

    def predict_cumulative_hazard(self, df, batch_size=512):
        '''Get cumulative hazards for dataset df.
        H(x, t) = H0(t) exp(g(x))

        Parameters:
            df: Pandas dataframe with covariates.
            batch_size: Batch size passed calculation of g_preds.

        Returns:
            Pandas data frame with cumulative hazards. One columns for
            each individual in the df.
        '''
        assert hasattr(self, 'baseline_cumulative_hazard_'), 'Need to fit model first.'
        bch = self.baseline_cumulative_hazard_.values.reshape(-1, 1)
        expg = self.predict_expg(df, batch_size, return_numpy=True).reshape(1, -1)
        return pd.DataFrame(bch.dot(expg), columns=df.index,
                            index=self.baseline_cumulative_hazard_.index)

    def predict_survival_function(self, df, batch_size=512):
        '''Predict survival function for dataset df.
        S(x, t) = exp(-H0(t)*exp(g(x))).

        Parameters:
            df: Pandas dataframe with covariates.
            batch_size: Batch size passed calculation of g_preds.

        Returns:
            Pandas data frame with survival functions. One columns for
            each individual in the df.
        '''
        return np.exp(-self.predict_cumulative_hazard(df, batch_size))
    
    def _lookup_baseline_cumulative_hazard(self, time):
        '''Search for baseline_cumulative_hazard at the times time.
        Parameters:
            time: Scalar or numpy array of times.
        '''
        bch_times = self.baseline_cumulative_hazard_.index.values
        idx = np.searchsorted(bch_times, time)
        exact = pd.Series(time).isin(bch_times).values
        idx -= (1 - exact)
        if any(idx < 0):
            warnings.warn('Give time values smaller than firs event.')
            idx[idx < 0] = 0
        bch = self.baseline_cumulative_hazard_.iloc[idx]
        return bch

    def predict_cumulative_hazard_at_times(self, times, df, batch_size=512, return_df=True):
        '''Predict cumulative hazard H(x, t) = exp(- H0(t)*exp(g(x))), only at given times.

        Parameters:
            times: Number or iterable with times.
            df: Pandas dataframe with covariates.
            batch_size: Batch size passed calculation of g_preds.
            return_df: Whether or not to return a pandas dataframe or a numpy matrix.

        Returns:
            Pandas dataframe (or numpy matrix) [len(times), len(df)] with cumulative hazard
            estimates.
        '''
        assert hasattr(self, 'baseline_cumulative_hazard_'), 'Need to fit model first.'
        if not hasattr(times, '__iter__'):
            times = [times]
        bch = self._lookup_baseline_cumulative_hazard(times).values.reshape(-1, 1)
        expg = self.predict_expg(df, batch_size, return_numpy=True).reshape(1, -1)
        res = bch.dot(expg)
        if return_df:
            return pd.DataFrame(res, index=times, columns=df.index)
        return res

    def predict_survival_at_times(self, times, df, batch_size=512, return_df=True):
        '''Predict survival S(x, t) = exp(- H0(t)*exp(g(x))), only at given times.

        Parameters:
            times: Iterable with times.
            df: Pandas dataframe with covariates.
            batch_size: Batch size passed calculation of g_preds.
            return_df: Whether or not to return a pandas dataframe or a numpy matrix.

        Returns:
            Pandas dataframe (or numpy matrix) [len(times), len(df)] with survival estimates.
        '''
        return np.exp(-self.predict_cumulative_hazard_at_times(times, df, batch_size, return_df))

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
        train_durations = self.df[self.duration_col].values
        train_events = self.df[self.event_col].values

        return brier_score(times, prob_alive, durations, events, train_durations,
                           train_events)

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
            return self.predict_survival_at_times(times, df, batch_size, False)

        durations = df[self.duration_col].values
        events = df[self.event_col].values
        train_durations = self.df[self.duration_col].values
        train_events = self.df[self.event_col].values

        return integrated_brier_score(prob_alive_func, durations, events, train_durations,
                                      train_events, times_grid, n_grid_points)

    def parameters(self):
        '''Returns an iterator over module parameters.
        This is typically passed to an optimizer.
        '''
        return self.g.parameters()

    @property
    def parameters_numpy(self):
        '''Returns parameters in net as a list of numpy tensors.'''
        raise NotImplementedError()

    def print_model(self):
        '''Prints the torch model'''
        print(self.g)


class CoxTime(CoxPH):
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
            n_workers=0, verbose=1, callbacks=None, compute_hazards=False):
        '''Fit the Cox Propertional Hazard model to a dataset, with time as a covariate.
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
            n_workers: Number of workers for preparing data.
            callbacks: List of callbacks.
            compute_hazards: If we should compute hazards when training has finished.

        # Returns:
        #     self, with additional properties: hazards_
        '''
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
        dataloader = self._make_dataloader(self.Xtr, self.time_fail, self.gr_alive,
                                           n_control, batch_size, n_workers)
        log = self.fit_dataloader(dataloader, epochs, verbose, callbacks)

        self.set_hazards(self.df, compute_hazards=compute_hazards)
        return log

    @staticmethod
    def _make_dataloader(Xtr, time_fail, gr_alive, n_control, batch_size, n_workers):
        assert Xtr.dtype == np.float32, 'Need Xtr to be np.float32'
        assert time_fail.dtype == 'float32', 'To use time as a covariate, we need dtype: float32'
        trainset = CoxPrepareWithTime(Xtr, time_fail, gr_alive, n_control)
        dataloader = DataLoaderBatch(trainset, batch_size=batch_size, shuffle=True,
                                     num_workers=n_workers)
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
        return super(CoxPH, self).predict_g(x, batch_size, return_numpy, eval_)

    def compute_baseline_hazard(self, df=None, max_duration=np.inf, sample=None, batch_size=512,
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
            Pandas series with baseline hazard. Index is duration_col.
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

        def compute_expg_at_risk(ix, t):
            expg = self.predict_expg(df.iloc[ix:].assign(**{self.duration_col: t}), batch_size)
            return expg.flatten().sum()

        df = df.sort_values(self.duration_col).reset_index(drop=True)
        self.baseline_hazards_df = df # Useful for e.g. integrated brier score
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
                     .rename('baseline_hazard'))
        if set_hazards:
            self.baseline_hazard_ = base_haz
            self.baseline_cumulative_hazard_ = self.compute_baseline_cumulative_hazard(baseline_hazard=base_haz)

        return base_haz

    def predict_cumulative_hazard(self, df, baseline_hazard_=None, batch_size=512, verbose=0):
        '''Get cumulative hazards for dataset df.
        H(x, t) = \sum [h0(t) exp(g(x, t))]

        Parameters:
            df: Pandas dataframe with covariates.
            baseline_hazard_: Pandas series with index: time, and values: baseline hazards.
            batch_size: Batch size passed calculation of g_preds.
            verbose: If we should print progress.

        Returns:
            Pandas data frame with cumulative hazards. One columns for
            each individual in the df.
        '''
        if baseline_hazard_ is None:
            assert hasattr(self, 'baseline_hazard_'), 'Need to fit model first.'
            assert self.baseline_hazard_ is not None, 'Need to compute baseline_hazard_'
            baseline_hazard_ = self.baseline_hazard_
        assert baseline_hazard_.index.is_monotonic_increasing,\
            'Need index of baseline_hazard_ to be monotonic increasing, as it represents time.'

        def expg_at_time(t):
            return self.predict_expg(df.assign(**{self.duration_col: t}), batch_size).flatten()

        r, c = baseline_hazard_.shape[0], df.shape[0]
        hazards = np.empty((r, c))
        for idx, t in enumerate(baseline_hazard_.index):
            if verbose:
                print(i, 'of', len(baseline_hazard_))
            hazards[idx, :] = expg_at_time(t)
        hazards *= baseline_hazard_.values.reshape(-1, 1)
        return pd.DataFrame(hazards, columns=df.index, index=baseline_hazard_.index).cumsum()

    def predict_survival_function(self, df, baseline_hazard_=None, batch_size=512, verbose=0):
        '''Predict survival function for dataset df.
        S(x, t) = exp(-H(x, t))

        Parameters:
            df: Pandas dataframe with covariates.
            baseline_hazard_: Pandas series with index: time, and values: baseline hazards.
            batch_size: Batch size passed calculation of g_preds.
            verbose: If we should print progress.

        Returns:
            Pandas data frame with survival functions. One columns for
            each individual in the df.
        '''
        return np.exp(-self.predict_cumulative_hazard(df, baseline_hazard_, batch_size, verbose))


    def predict_cumulative_hazard_at_times(self, times, df, baseline_hazard_=None, batch_size=512, 
                                           return_df=True, verbose=0):
        '''Predict cumulative hazard only at given times. This is not as efficient as
        for the proportional hazards models.

        Parameters:
            times: Number or iterable with times.
            df: Pandas dataframe with covariates.
            baseline_hazard_: Pandas series with index: time, and values: baseline hazards.
            batch_size: Batch size passed calculation of g_preds.
            return_df: Whether or not to return a pandas dataframe or a numpy matrix.
            verbose: If we should print progress.

        Returns:
            Pandas dataframe (or numpy matrix) [len(times), len(df)] with cumulative hazard
            estimates.
        '''
        assert hasattr(self, 'baseline_hazard_'), 'Need to fit model first.'
        if not hasattr(times, '__iter__'):
            times = [times]
        cum_haz = self.predict_cumulative_hazard(df, baseline_hazard_, batch_size, verbose)
        times_idx = search_sorted_idx(cum_haz.index.values, times)
        cum_haz = cum_haz.iloc[times_idx]
        if return_df:
            return cum_haz
        return cum_haz.as_matrix()

    def predict_survival_at_times(self, times, df, baseline_hazard_=None, batch_size=512,
                                 return_df=True, verbose=0):
        '''Predict survival function at given times.
        Not very efficient!!!

        Parameters:
            times: Iterable with times.
            df: Pandas dataframe with covariates.
            baseline_hazard_: Pandas series with index: time, and values: baseline hazards.
            batch_size: Batch size passed calculation of g_preds.
            return_df: Whether or not to return a pandas dataframe or a numpy matrix.
            verbose: If we should print progress.

        Returns:
            Pandas dataframe (or numpy matrix) [len(times), len(df)] with survival estimates.
        '''
        return np.exp(-self.predict_cumulative_hazard_at_times(times, df, baseline_hazard_, batch_size, return_df,
                                                               verbose))

    def concordance_index(self, df, g_preds=None, batch_size=256):
        raise NotImplementedError()

    def partial_log_likelihood(self, df, g_preds=None, batch_size=512):
        raise NotImplementedError()


def search_sorted_idx(array, values):
    '''For sorted array, get index of values.
    If value not in array, give left index of value.
    '''
    idx = np.searchsorted(array, values)
    not_exact = values != array[idx]
    idx -= not_exact
    if any(idx < 0):
        warnings.warn('Given value smaller than first value')
        idx[idx < 0] = 0
    return idx


class CoxPHLinear(CoxPH):
    '''This class implements Cox's proportional hazard model:
    h(t|x) = h_0(t)*exp(g(x)), where g(x) = beta^T x.

    Parameters:
        input_size: Size of x, i.e. number of covariates.
        set_optim_func: Function for setting pytorch optimizer.
            If None optimizer is set to Adam with default parameters.
            Function should take one argument (pytorch model) and return the optimizer.
            See CoxPHLinear.set_optim_default as an example.
        cuda: Set to True if use GPU.
    '''
    def __init__(self, input_size, set_optim_func=None, cuda=False):
        self.input_size = input_size
        g_model = self._make_g_model(self.input_size)
        self.set_optim_func = set_optim_func
        if self.set_optim_func is None:
            self.set_optim_func = self.set_optim_default
        optimizer = self.set_optim_func(g_model)
        super().__init__(g_model, optimizer, cuda)

    @staticmethod
    def set_optim_default(g_model):
        return optim.Adam(g_model.parameters())

    def _make_g_model(self, input_size):
        return nn.Sequential(nn.Linear(input_size, 1, bias=False))

    def fit(self, df, duration_col, event_col=None, n_control=1, batch_size=64, epochs=500,
            n_workers=0, verbose=1, strata=None, callbacks=None, compute_hazards=True,
            early_stopping=True):
        '''Fit the Cox Propertional Hazard model to a dataset. Tied survival times
        are handled using Efron's tie-method.

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
            n_workers: Number of workers for preparing data.
            strata: Specify a list of columns to use in stratification. This is useful if a
                catagorical covariate does not obey the proportional hazard assumption. This
                is used similar to the `strata` expression in R.
                See http://courses.washington.edu/b515/l17.pdf.
            callbacks: List of callbacks.
            compute_hazards: If we should compute hazards when training has finished.
            early_stopping: Use prespesifed early stopping callback to stop when loss hasn't
                imporved for last 5 epochs.

        # Returns:
        #     self, with additional properties: hazards_
        '''
        if callbacks is None:
            callbacks = []
        if early_stopping:
            callbacks.append(EarlyStoppingTrainLoss())
        return super().fit(df, duration_col, event_col, n_control, batch_size, epochs,
                           n_workers, verbose, strata, callbacks, compute_hazards)


class CoxPHMLP(CoxPH):
    '''This class implements Cox's proportional hazard model:
    h(t|x) = h_0(t)*exp(g(x)), where g(x) = is an one-hidden-layer MLP with elu activation.

    Parameters:
        input_size: Size of x, i.e. number of covariates.
        hidden_size: Size of hidden layer.
        set_optim_func: Function for setting pytorch optimizer.
            If None optimizer is set to SGD with lr=0.01, and momentum=0.9.
            Function should take one argument (pytorch model) and return the optimizer.
            See Cox.set_optim_default as an example.
        cuda: Set to True if use GPU.
    '''
    def __init__(self, input_size, hidden_size, set_optim_func=None, cuda=False):
        self.input_size = input_size
        self.hidden_size = hidden_size
        g_model = self._make_g_model(self.input_size, self.hidden_size)
        self.set_optim_func = set_optim_func
        if self.set_optim_func is None:
            self.set_optim_func = self.set_optim_default
        optimizer = self.set_optim_func(g_model)
        super().__init__(g_model, optimizer, cuda)

    @staticmethod
    def set_optim_default(g_model):
        return optim.SGD(g_model.parameters(), lr=0.01, momentum=0.9)

    def _make_g_model(self, input_size, hidden_size):
        return nn.Sequential(nn.Linear(input_size, hidden_size), nn.ELU(),
                             nn.Linear(hidden_size, 1, bias=False))

    def fit(self, df, duration_col, event_col=None, n_control=1, batch_size=64, epochs=500,
            n_workers=0, verbose=1, strata=None, callbacks=None, compute_hazards=True,
            early_stopping=True):
    # def fit(self, df, duration_col, event_col=None, batch_size=64, epochs=500,
    #         num_workers=0, n_control=1, verbose=1, strata=None, early_stopping=True,
    #         callbacks=None):
        '''Fit the Cox Propertional Hazard model to a dataset. Tied survival times
        are handled using Efron's tie-method.

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
            n_workers: Number of workers for preparing data.
            strata: Specify a list of columns to use in stratification. This is useful if a
                catagorical covariate does not obey the proportional hazard assumption. This
                is used similar to the `strata` expression in R.
                See http://courses.washington.edu/b515/l17.pdf.
            callbacks: List of callbacks.
            compute_hazards: If we should compute hazards when training has finished.
            early_stopping: Use prespesifed early stopping callback to stop when loss hasn't
                imporved for last 5 epochs.

        # Returns:
        #     self, with additional properties: hazards_
        '''
        if callbacks is None:
            callbacks = []
        if early_stopping:
            callbacks.append(EarlyStoppingTrainLoss())
        return super().fit(df, duration_col, event_col, n_control, batch_size, epochs,
                           n_workers, verbose, strata, callbacks, compute_hazards)


class FuncTorch(nn.Module):
    def __init__(self, func):
        self.func = func
        super().__init__()

    def forward(self, x):
        return self.func(x)


class CoxFunc(CoxPH):
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

    def _fake_fit(self, df, duration_col, event_col):
        return super().fit(df, duration_col, event_col, epochs=0)

    def fit(self, *args, **kwargs):
        '''It's not possible to fit this object.'''
        raise ValueError("It's not possible to fit this object")


class CoxLifelines(CoxFunc):
    '''Class for doing same as CoxNN on lifelines cph object.

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
