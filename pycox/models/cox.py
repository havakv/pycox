import os
import warnings
import numpy as np
import pandas as pd
import torch
import torchtuples as tt
from pycox import models

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


class _CoxBase(models.base.SurvBase):
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
            batch_size {int} -- Elements in each batch (default: {256})
            epochs {int} -- Number of epochs (default: {1})
            callbacks {list} -- list of callbacks (default: {None})
            verbose {bool} -- Print progress (default: {True})
            num_workers {int} -- Number of workers used in the dataloader (default: {0})
            shuffle {bool} -- If we should shuffle the order of the dataset (default: {True})
            **kwargs are passed to 'make_dataloader' method.
    
        Returns:
            TrainingLogger -- Training log
        """
        self.training_data = tt.tuplefy(input, target)
        return super().fit(input, target, batch_size, epochs, callbacks, verbose,
                           num_workers, shuffle, metrics, val_data, val_batch_size,
                           **kwargs)

    def _compute_baseline_hazards(self, input, df, max_duration, batch_size, eval_=True, num_workers=0):
        raise NotImplementedError

    def target_to_df(self, target):
        durations, events = tt.tuplefy(target).to_numpy()
        df = pd.DataFrame({self.duration_col: durations, self.event_col: events}) 
        return df

    def compute_baseline_hazards(self, input=None, target=None, max_duration=None, sample=None, batch_size=8224,
                                set_hazards=True, eval_=True, num_workers=0):
        """Computes the Breslow estimates form the data defined by `input` and `target`
        (if `None` use training data).

        Typically call
        model.compute_baseline_hazards() after fitting.
        
        Keyword Arguments:
            input  -- Input data (train input) (default: {None})
            target  -- Target data (train target) (default: {None})
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
        input = tt.tuplefy(input).to_numpy().iloc[df.index.values]
        base_haz = self._compute_baseline_hazards(input, df, max_duration, batch_size,
                                                  eval_=eval_, num_workers=num_workers)
        if set_hazards:
            self.compute_baseline_cumulative_hazards(set_hazards=True, baseline_hazards_=base_haz)
        return base_haz

    def compute_baseline_cumulative_hazards(self, input=None, target=None, max_duration=None, sample=None,
                                            batch_size=8224, set_hazards=True, baseline_hazards_=None,
                                            eval_=True, num_workers=0):
        """See `compute_baseline_hazards. This is the cumulative version."""
        if ((input is not None) or (target is not None)) and (baseline_hazards_ is not None):
            raise ValueError("'input', 'target' and 'baseline_hazards_' can not both be different from 'None'.")
        if baseline_hazards_ is None:
            baseline_hazards_ = self.compute_baseline_hazards(input, target, max_duration, sample, batch_size,
                                                             set_hazards=False, eval_=eval_, num_workers=num_workers)
        assert baseline_hazards_.index.is_monotonic_increasing,\
            'Need index of baseline_hazards_ to be monotonic increasing, as it represents time.'
        bch = (baseline_hazards_
                .cumsum()
                .rename('baseline_cumulative_hazards'))
        if set_hazards:
            self.baseline_hazards_ = baseline_hazards_
            self.baseline_cumulative_hazards_ = bch
        return bch

    def predict_cumulative_hazards(self, input, max_duration=None, batch_size=8224, verbose=False,
                                   baseline_hazards_=None, eval_=True, num_workers=0):
        """See `predict_survival_function`."""
        if type(input) is pd.DataFrame:
            input = self.df_to_input(input)
        if baseline_hazards_ is None:
            if not hasattr(self, 'baseline_hazards_'):
                raise ValueError('Need to compute baseline_hazards_. E.g run `model.compute_baseline_hazards()`')
            baseline_hazards_ = self.baseline_hazards_
        assert baseline_hazards_.index.is_monotonic_increasing,\
            'Need index of baseline_hazards_ to be monotonic increasing, as it represents time.'
        return self._predict_cumulative_hazards(input, max_duration, batch_size, verbose, baseline_hazards_,
                                                eval_, num_workers=num_workers)

    def _predict_cumulative_hazards(self, input, max_duration, batch_size, verbose, baseline_hazards_,
                                    eval_=True, num_workers=0):
        raise NotImplementedError

    def predict_surv_df(self, input, max_duration=None, batch_size=8224, verbose=False, baseline_hazards_=None,
                        eval_=True, num_workers=0):
        """Predict survival function for `input`. S(x, t) = exp(-H(x, t))
        Require computed baseline hazards.

        Arguments:
            input {np.array, tensor or tuple} -- Input x passed to net.

        Keyword Arguments:
            max_duration {float} -- Don't compute estimates for duration higher (default: {None})
            batch_size {int} -- Batch size (default: {8224})
            baseline_hazards_ {pd.Series} -- Baseline hazards. If `None` used `model.baseline_hazards_` (default: {None})
            eval_ {bool} -- If 'True', use 'eval' mode on net. (default: {True})
            num_workers {int} -- Number of workers in created dataloader (default: {0})

        Returns:
            pd.DataFrame -- Survival estimates. One columns for each individual.
        """
        return np.exp(-self.predict_cumulative_hazards(input, max_duration, batch_size, verbose, baseline_hazards_,
                                                       eval_, num_workers))

    def predict_surv(self, input, max_duration=None, batch_size=8224, numpy=None, verbose=False,
                     baseline_hazards_=None, eval_=True, num_workers=0):
        """Predict survival function for `input`. S(x, t) = exp(-H(x, t))
        Require compueted baseline hazards.

        Arguments:
            input {np.array, tensor or tuple} -- Input x passed to net.

        Keyword Arguments:
            max_duration {float} -- Don't compute estimates for duration higher (default: {None})
            batch_size {int} -- Batch size (default: {8224})
            numpy {bool} -- 'False' gives tensor, 'True' gives numpy, and None give same as input
                (default: {None})
            baseline_hazards_ {pd.Series} -- Baseline hazards. If `None` used `model.baseline_hazards_` (default: {None})
            eval_ {bool} -- If 'True', use 'eval' mode on net. (default: {True})
            num_workers {int} -- Number of workers in created dataloader (default: {0})

        Returns:
            pd.DataFrame -- Survival estimates. One columns for each individual.
        """
        surv = self.predict_surv_df(input, max_duration, batch_size, verbose, baseline_hazards_,
                                    eval_, num_workers)
        surv = torch.from_numpy(surv.values.transpose())
        return tt.utils.array_or_tensor(surv, numpy, input)

    def save_net(self, path, **kwargs):
        """Save self.net and baseline hazards to file.

        Arguments:
            path {str} -- Path to file.
            **kwargs are passed to torch.save

        Returns:
            None
        """
        path, extension = os.path.splitext(path)
        if extension == "":
            extension = '.pt'
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
        path, extension = os.path.splitext(path)
        if extension == "":
            extension = '.pt'
        super().load_net(path+extension, **kwargs)
        blh_path = path+'_blh.pickle'
        if os.path.isfile(blh_path):
            self.baseline_hazards_ = pd.read_pickle(blh_path)
            self.baseline_cumulative_hazards_ = self.baseline_hazards_.cumsum()

    def df_to_input(self, df):
        input = df[self.input_cols].values
        return input
    

class _CoxPHBase(_CoxBase):
    def _compute_baseline_hazards(self, input, df_target, max_duration, batch_size, eval_=True, num_workers=0):
        if max_duration is None:
            max_duration = np.inf

        # Here we are computing when expg when there are no events.
        #   Could be made faster, by only computing when there are events.
        return (df_target
                .assign(expg=np.exp(self.predict(input, batch_size, True, eval_, num_workers=num_workers)))
                .groupby(self.duration_col)
                .agg({'expg': 'sum', self.event_col: 'sum'})
                .sort_index(ascending=False)
                .assign(expg=lambda x: x['expg'].cumsum())
                .pipe(lambda x: x[self.event_col]/x['expg'])
                .fillna(0.)
                .iloc[::-1]
                .loc[lambda x: x.index <= max_duration]
                .rename('baseline_hazards'))

    def _predict_cumulative_hazards(self, input, max_duration, batch_size, verbose, baseline_hazards_,
                                    eval_=True, num_workers=0):
        max_duration = np.inf if max_duration is None else max_duration
        if baseline_hazards_ is self.baseline_hazards_:
            bch = self.baseline_cumulative_hazards_
        else:
            bch = self.compute_baseline_cumulative_hazards(set_hazards=False, 
                                                           baseline_hazards_=baseline_hazards_)
        bch = bch.loc[lambda x: x.index <= max_duration]
        expg = np.exp(self.predict(input, batch_size, True, eval_, num_workers=num_workers)).reshape(1, -1)
        return pd.DataFrame(bch.values.reshape(-1, 1).dot(expg), 
                            index=bch.index)

    def partial_log_likelihood(self, input, target, g_preds=None, batch_size=8224, eps=1e-7, eval_=True,
                               num_workers=0):
        '''Calculate the partial log-likelihood for the events in datafram df.
        This likelihood does not sample the controls.
        Note that censored data (non events) does not have a partial log-likelihood.

        Arguments:
            input {tuple, np.ndarray, or torch.tensor} -- Input to net.
            target {tuple, np.ndarray, or torch.tensor} -- Target labels.

        Keyword Arguments:
            g_preds {np.array} -- Predictions from `model.predict` (default: {None})
            batch_size {int} -- Batch size (default: {8224})
            eval_ {bool} -- If 'True', use 'eval' mode on net. (default: {True})
            num_workers {int} -- Number of workers in created dataloader (default: {0})

        Returns:
            Partial log-likelihood.
        '''
        df = self.target_to_df(target)
        if g_preds is None:
            g_preds = self.predict(input, batch_size, True, eval_, num_workers=num_workers)
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


class CoxPH(_CoxPHBase):
    """Cox proportional hazards model parameterized with a neural net.
    This is essentially the DeepSurv method [1].

    The loss function is not quite the partial log-likelihood, but close.    
    The difference is that for tied events, we use a random order instead of 
    including all individuals that had an event at that point in time.

    Arguments:
        net {torch.nn.Module} -- A pytorch net.
    
    Keyword Arguments:
        optimizer {torch or torchtuples optimizer} -- Optimizer (default: {None})
        device {str, int, torch.device} -- Device to compute on. (default: {None})
            Preferably pass a torch.device object.
            If 'None': use default gpu if available, else use cpu.
            If 'int': used that gpu: torch.device('cuda:<device>').
            If 'string': string is passed to torch.device('string').

    [1] Jared L. Katzman, Uri Shaham, Alexander Cloninger, Jonathan Bates, Tingting Jiang, and Yuval Kluger.
        Deepsurv: personalized treatment recommender system using a Cox proportional hazards deep neural network.
        BMC Medical Research Methodology, 18(1), 2018.
        https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/s12874-018-0482-1
    """
    def __init__(self, net, optimizer=None, device=None, loss=None):
        if loss is None:
            loss = models.loss.CoxPHLoss()
        super().__init__(net, loss, optimizer, device)


class CoxPHSorted(_CoxPHBase):
    """Cox proportional hazards model parameterized with a neural net.
    This is essentially the DeepSurv method [1].

    The loss function is not quite the partial log-likelihood, but close.    
    The difference is that for tied events, we use a random order instead of 
    including all individuals that had an event at that point in time.

    Arguments:
        net {torch.nn.Module} -- A pytorch net.
    
    Keyword Arguments:
        optimizer {torch or torchtuples optimizer} -- Optimizer (default: {None})
        device {str, int, torch.device} -- Device to compute on. (default: {None})
            Preferably pass a torch.device object.
            If 'None': use default gpu if available, else use cpu.
            If 'int': used that gpu: torch.device('cuda:<device>').
            If 'string': string is passed to torch.device('string').

    [1] Jared L. Katzman, Uri Shaham, Alexander Cloninger, Jonathan Bates, Tingting Jiang, and Yuval Kluger.
        Deepsurv: personalized treatment recommender system using a Cox proportional hazards deep neural network.
        BMC Medical Research Methodology, 18(1), 2018.
        https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/s12874-018-0482-1
    """
    def __init__(self, net, optimizer=None, device=None, loss=None):
        warnings.warn('Use `CoxPH` instead. This will be removed', DeprecationWarning)
        if loss is None:
            loss = models.loss.CoxPHLossSorted()
        super().__init__(net, loss, optimizer, device)

    @staticmethod
    def make_dataloader(data, batch_size, shuffle, num_workers=0):
        dataloader = tt.make_dataloader(data, batch_size, shuffle, num_workers,
                                        make_dataset=models.data.DurationSortedDataset)
        return dataloader

    def make_dataloader_predict(self, input, batch_size, shuffle=False, num_workers=0):
        dataloader = super().make_dataloader(input, batch_size, shuffle, num_workers)
        return dataloader
