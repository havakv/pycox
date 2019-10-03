import numpy as np
import pandas as pd
import torchtuples as tt
from pycox.preprocessing.label_transforms import LabTransCoxTime
from pycox import models


class _CoxCCBase(models.cox._CoxBase):
    make_dataset = NotImplementedError

    def __init__(self, net, optimizer=None, device=None, shrink=0.):
        super().__init__(net, self.make_loss(shrink), optimizer, device)

    def make_loss(self, shrink):
        return models.loss.CoxCCLoss(shrink)

    def fit(self, input, target, batch_size=256, epochs=1, callbacks=None, verbose=True,
            num_workers=0, shuffle=True, metrics=None, val_data=None, val_batch_size=8224,
            n_control=1, shrink=None, **kwargs):
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
            n_control {int} -- Number of control samples.
            **kwargs are passed to 'make_dataloader' method.
    
        Returns:
            TrainingLogger -- Training log
        """
        input, target = self._sorted_input_target(input, target)
        if shrink is not None:
            self.loss.shrink = shrink
        return super().fit(input, target, batch_size, epochs, callbacks, verbose,
                           num_workers, shuffle, metrics, val_data, val_batch_size,
                           n_control=n_control, **kwargs)

    def compute_metrics(self, input, target, metrics):
        if (self.loss is None) and (self.loss in metrics.values()):
            raise RuntimeError(f"Need to specify a loss (self.loss). It's currently None")
        assert target is None, 'Need target to be none, input=(case, control)'
        input = self._to_device(input)
        batch_size = input.lens().flatten().get_if_all_equal()
        if batch_size is None:
            raise RuntimeError("All elements in input does not have the same lenght.")
        case, control = input # both are TupleTree
        input_all = tt.TupleTree((case,) + control).cat()
        g_all = self.net(*input_all)
        g_all = tt.tuplefy(g_all).split(batch_size).flatten()
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
        input, target = self._sorted_input_target(*data)
        durations, events = target
        dataset = self.make_dataset(input, durations, events, n_control)
        dataloader = tt.data.DataLoaderSlice(dataset, batch_size=batch_size,
                                             shuffle=shuffle, num_workers=num_workers)
        return dataloader

    @staticmethod
    def _sorted_input_target(input, target):
        durations, _ = target#.to_numpy()
        idx_sort = np.argsort(durations)
        if (idx_sort == np.arange(0, len(idx_sort))).all():
            return input, target
        input = tt.tuplefy(input).iloc[idx_sort]
        target = tt.tuplefy(target).iloc[idx_sort]
        return input, target


class CoxCC(_CoxCCBase, models.cox._CoxPHBase):
    """Cox proportional hazards model parameterized with a neural net and
    trained with case-control sampling.
    This is similar to DeepSurv, but use an approximation of the loss function.
    
    Arguments:
        net {torch.nn.Module} -- A pytorch net.
    
    Keyword Arguments:
        optimizer {torch or torchtuples optimizer} -- Optimizer (default: {None})
        device {string, int, or torch.device} -- See torchtuples.Model (default: {None})
    """
    make_dataset = models.data.CoxCCDataset


class CoxTime(_CoxCCBase):
    """A Cox model that does not have proportional hazards, trained with case-control sampling.
    Se paper for explanation http://jmlr.org/papers/volume20/18-424/18-424.pdf
    
    Arguments:
        net {torch.nn.Module} -- A pytorch net.
    
    Keyword Arguments:
        optimizer {torch or torchtuples optimizer} -- Optimizer (default: {None})
        device {string, int, or torch.device} -- See torchtuples.Model (default: {None})
    """
    make_dataset = models.data.CoxTimeDataset
    label_transform = LabTransCoxTime

    def __init__(self, net, optimizer=None, device=None, shrink=0., labtrans=None):
        self.labtrans = labtrans
        super().__init__(net, optimizer, device, shrink)

    def make_dataloader_predict(self, input, batch_size, shuffle=False, num_workers=0):
        input, durations = input
        input = tt.tuplefy(input)
        durations = tt.tuplefy(durations)
        new_input = input + durations 
        dataloader = super().make_dataloader_predict(new_input, batch_size, shuffle, num_workers)
        return dataloader


    def predict_surv_df(self, input, max_duration=None, batch_size=8224, verbose=False, baseline_hazards_=None,
                        eval_=True, num_workers=0):
        surv = super().predict_surv_df(input, max_duration, batch_size, verbose, baseline_hazards_,
                                       eval_, num_workers)
        if self.labtrans is not None:
            surv.index = self.labtrans.map_scaled_to_orig(surv.index)
        return surv

    def compute_baseline_hazards(self, input=None, target=None, max_duration=None, sample=None, batch_size=8224,
                                set_hazards=True, eval_=True, num_workers=0):
        if (input is None) and (target is None):
            if not hasattr(self, 'training_data'):
                raise ValueError('Need to fit, or supply a input and target to this function.')
            input, target = self.training_data
        df = self.target_to_df(target)
        if sample is not None:
            if sample >= 1:
                df = df.sample(n=sample)
            else:
                df = df.sample(frac=sample)
            df = df.sort_values(self.duration_col)
        input = tt.tuplefy(input).to_numpy().iloc[df.index.values]
        base_haz = self._compute_baseline_hazards(input, df, max_duration, batch_size, eval_, num_workers)
        if set_hazards:
            self.compute_baseline_cumulative_hazards(set_hazards=True, baseline_hazards_=base_haz)
        return base_haz

    def _compute_baseline_hazards(self, input, df_train_target, max_duration, batch_size, eval_=True,
                                  num_workers=0):
        if max_duration is None:
            max_duration = np.inf
        def compute_expg_at_risk(ix, t):
            sub = input.iloc[ix:]
            n = sub.lens().flatten().get_if_all_equal()
            t = np.repeat(t, n).reshape(-1, 1).astype('float32')
            return np.exp(self.predict((sub, t), batch_size, True, eval_, num_workers=num_workers)).flatten().sum()

        if not df_train_target[self.duration_col].is_monotonic_increasing:
            raise RuntimeError(f"Need 'df_train_target' to be sorted by {self.duration_col}")
        input = tt.tuplefy(input)
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

    def _predict_cumulative_hazards(self, input, max_duration, batch_size, verbose, baseline_hazards_,
                                    eval_=True, num_workers=0):
        def expg_at_time(t):
            t = np.repeat(t, n_cols).reshape(-1, 1).astype('float32')
            return np.exp(self.predict((input, t), batch_size, True, eval_, num_workers=num_workers)).flatten()

        input = tt.tuplefy(input)
        max_duration = np.inf if max_duration is None else max_duration
        baseline_hazards_ = baseline_hazards_.loc[lambda x: x.index <= max_duration]
        n_rows, n_cols = baseline_hazards_.shape[0], input.lens().flatten().get_if_all_equal()
        hazards = np.empty((n_rows, n_cols))
        for idx, t in enumerate(baseline_hazards_.index):
            if verbose:
                print(idx, 'of', len(baseline_hazards_))
            hazards[idx, :] = expg_at_time(t)
        hazards[baseline_hazards_.values == 0] = 0.  # in case hazards are inf here
        hazards *= baseline_hazards_.values.reshape(-1, 1)
        return pd.DataFrame(hazards, index=baseline_hazards_.index).cumsum()

    def partial_log_likelihood(self, input, target, batch_size=8224, eval_=True, num_workers=0):
        def expg_sum(t, i):
            sub = input_sorted.iloc[i:]
            n = sub.lens().flatten().get_if_all_equal()
            t = np.repeat(t, n).reshape(-1, 1).astype('float32')
            return np.exp(self.predict((sub, t), batch_size, True, eval_, num_workers=num_workers)).flatten().sum()

        durations, events = target
        df = pd.DataFrame({self.duration_col: durations, self.event_col: events})
        df = df.sort_values(self.duration_col)
        input = tt.tuplefy(input)
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
        g_preds = self.predict((input_event, durations_event), batch_size, True, eval_, num_workers=num_workers).flatten()
        pll = (pll
               .assign(_g_preds=g_preds)
               .reset_index()
               .merge(times, on=self.duration_col)
               .set_index(idx_name)
               .assign(pll=lambda x: x['_g_preds'] - np.log(x['_expg_sum']))
               ['pll'])

        pll.index.name = idx_name_old
        return pll
