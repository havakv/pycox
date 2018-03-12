'''
Callbacks.
'''
import warnings
import time
import numpy as np
import pandas as pd
from sklearn import metrics
from tqdm import trange
import torch
from torch import optim
from torch.autograd import Variable


class CallbacksList(object):
    '''Object for holding all callbacks.

    Parameters:
        callbacks_list: List containing callback objects.
    '''
    def __init__(self, callbacks=None):
        self.callbacks = callbacks if callbacks else []

    def add(self, callback):
        self.callbacks.append(callback)

    def give_model(self, model):
        for c in self.callbacks:
            c.give_model(model)

    def on_fit_start(self):
        for c in self.callbacks:
            c.on_fit_start()

    def before_step(self):
        stop_signal = False
        for c in self.callbacks:
            stop_signal += c.before_step()
        return stop_signal

    def on_batch_end(self):
        for c in self.callbacks:
            c.on_batch_end()

    def on_epoch_end(self):
        stop_signal = False
        for c in self.callbacks:
            stop_signal += c.on_epoch_end()
        return stop_signal


class Callback(object):
    '''Abstract class for callbacks.
    '''
    def give_model(self, model):
        self.model = model

    def on_fit_start(self):
        pass

    def before_step(self):
        stop_signal = False
        return stop_signal

    def on_batch_end(self):
        pass

    def on_epoch_end(self):
        stop_signal = False
        return stop_signal


class TrainingLogger(Callback):
    '''Holding statistics about training.'''
    def __init__(self, verbose=1):
        self.epoch = 0
        self.epochs = []
        self.loss = []
        self._verbose = verbose

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        self._verbose = value

    def on_fit_start(self):
        self.prev_time = time.time()
        self.batch_loss = []
        if self.verbose == 2:
            self._make_prog_bar()

    def _make_prog_bar(self):
        self.prog_bar = trange(self.model.fit_info['batches_per_epoch'],
                               desc=str(self.epoch))
        self.prog_bar = iter(self.prog_bar)

    def _get_loss(self):
        return self.model.batch_loss.data[0]

    def on_batch_end(self):
        loss = self._get_loss()
        self.batch_loss.append(loss)
        if self.verbose == 2:
            next(self.prog_bar)

    def on_epoch_end(self):
        loss = np.mean(self.batch_loss)
        self.epochs.append(self.epoch)
        self.loss.append(loss)
        # if (self.verbose == 1) or (self.verbose == 2):
        if self.verbose:
            self.print_on_epoch_end()
        self.epoch += 1
        self.batch_loss = []
        if self.verbose == 2:
            self._make_prog_bar()
        return False
    
    def get_measures(self):
        string = '\tloss: %.4f,' % self.loss[-1]
        if self.verbose.__class__ is dict:
            for name, mm in self.verbose.items():
                string += '\t%s:' % name
                for sc in mm.scores:
                    string += ' %.4f,' % sc[-1]
        return string[:-1]

    def print_on_epoch_end(self):
        new_time = time.time()
        # loss = self.loss[-1]
        string = 'Epoch: %d,\ttime: %d sec,' % (self.epoch, new_time - self.prev_time)
        print(string + self.get_measures())
        self.prev_time = new_time
    # def print_on_epoch_end(self, loss):
    #     new_time = time.time()
    #     print('Epoch: %d,\ttime: %d sec,\tloss: %.4f'
    #           % (self.epoch, new_time - self.prev_time, loss))
    #     self.prev_time = new_time

    def history(self, df=True):
        '''df: if True returns pd.DataFrame, else dict.
        '''
        history = dict(epoch=self.epochs, loss=self.loss)
        if df:
            return pd.DataFrame(history).set_index('epoch')
        return history

    def to_pandas(self):
        return self.history()


class EarlyStoppingTrainLoss(Callback):
    '''Stop trainig when the training loss has stopped improving.

    For stopping using the full partial log-likelihood, (on train or val)
    see EarlyStopping.

    Parameters:
        min_delta: Minimum change in the monitored quantity to qualify as an improvement,
            i.e. an absolute change of less than min_delta, will count as no improvement.
        patience: Number of epochs with no improvement after which training will be stopped.
    '''
    def __init__(self, min_delta=0, patience=5):
        warnings.warn('Shoud not use EarlyStoppingTrainLoss!!! Revrite to early stopping.')
        self.min_delta = min_delta
        self.patience = patience
        self.val = np.inf
        self.n = 0
        self.loss = []

    def on_fit_start(self):
        self.batch_loss = []

    def on_batch_end(self):
        self.batch_loss.append(self.model.batch_loss.data[0])

    def on_epoch_end(self):
        loss = np.mean(self.batch_loss)
        self.loss.append(loss)
        if loss < (self.val - self.min_delta):
            self.val = loss
            self.n = -1
        self.n += 1
        self.batch_loss = []
        stop_signal = True if self.n >= self.patience else False
        return stop_signal


class EarlyStopping(Callback):
    '''Stop training when monitored quantity has stopped improving.
    Takes a MonitorMetrics object and runs it as a callback.
    Use first metric in mm_obj to determine early stopping.

    Parameters:
        mm_obj: MonitorMetrics object, where first metric is used for early stopping.
            E.g. MonitorMetricsSurvival(df_val, 'cindex').
        minimize: If we are to minimize or maximize monitor.
        min_delta: Minimum change in the monitored quantity to qualify as an improvement,
            i.e. an absolute change of less than min_delta, will count as no improvement.
        patience: Number of epochs with no improvement after which training will be stopped.
        model_file_path: If spesified, the model weights will be stored whever a better score
            is achieved.
    '''
    def __init__(self, mm_obj, minimize=False, min_delta=0, patience=5, model_file_path=None):
        self.mm_obj = mm_obj
        self.minimize = minimize
        self.min_delta = min_delta
        self.patience = patience
        self.model_file_path = model_file_path
        self.val = np.inf if self.minimize else -np.inf
        self.scores = []
        self.n = 0

    def give_model(self, model):
        super().give_model(model)
        self.mm_obj.give_model(model)

    def on_fit_start(self):
        self.mm_obj.on_fit_start()

    def before_step(self):
        return self.mm_obj.before_step()

    def on_batch_end(self):
        self.mm_obj.on_batch_end()

    def on_epoch_end(self):
        self.mm_obj.on_epoch_end()
        score = self.mm_obj.scores[0][-1]
        self.scores.append(score)

        if self.minimize:
            if score < (self.val - self.min_delta):
                self.val = score
                self.n = -1
        else:
            if score > (self.val + self.min_delta):
                self.val = score
                self.n = -1
        self.n += 1

        if (self.n == 0) and (self.model_file_path is not None):
            self.model.save_model_weights(self.model_file_path)

        stop_signal = True if self.n >= self.patience else False
        return stop_signal


class MonitorMetricsBase(Callback):
    '''Abstract class for monitoring metrics during training progress.

    Need to implement 'get_score_args' function to make it work.
    See MonitorMetricsXy for an example.

    Parameters:
        monitor_funcs: Function, list, or dict of functions giving quiatities that should
            be monitored.
            The function takes argumess (df, preds) and should return a score.
        batch_size: Batch size used for calculating the scores.
    '''
    def __init__(self, monitor_funcs, per_epoch=1, batch_size=512,):
        if monitor_funcs.__class__ is dict:
            self.monitor_names = list(monitor_funcs.keys())
            self.monitor_funcs = monitor_funcs.values()
        elif monitor_funcs.__class__ == list:
            self.monitor_names = list(range(len(monitor_funcs)))
            self.monitor_funcs = monitor_funcs
        else:
            self.monitor_names = ['monitor']
            self.monitor_funcs = [monitor_funcs]

        self.per_epoch = per_epoch
        self.batch_size = batch_size
        self.epoch = 0
        self.scores = [[] for _ in self.monitor_funcs]
        self.epochs = []

    def get_score_args(self):
        '''This function should create arguments to the monitor function.
        Typically it can return a tuple with (y_true, preds), to calculate e.g. auc.
        '''
        raise NotImplementedError('Need to implement this method!')

    def on_epoch_end(self):
        if self.epoch % self.per_epoch != 0:
            self.epoch += 1
            return False

        data, preds = self.get_score_args()
        for score_list, mon_func in zip(self.scores, self.monitor_funcs):
            score_list.append(mon_func(data, preds))

        self.epochs.append(self.epoch)
        self.epoch += 1
        return False

    def to_pandas(self):
        '''Return scores as a pandas dataframe'''
        scores = np.array(self.scores).transpose()
        return (pd.DataFrame(scores, columns=self.monitor_names)
                .set_index(np.array(self.epochs)))


class MonitorTrainLoss(MonitorMetricsBase):
    '''Monitor metrics for training loss.

    Parameters:
        per_epoch: How often to calculate.
    '''
    def __init__(self, per_epoch=1):
        monitor_funcs = {'train_loss': self.get_loss}
        super().__init__(monitor_funcs, per_epoch, None)
    
    def get_loss(self, *args, **kwargs):
        loss = np.mean(self.batch_loss)
        return loss

    def get_score_args(self):
        return None, None

    def on_fit_start(self):
        self.batch_loss = []

    def on_batch_end(self):
        self.batch_loss.append(self.model.batch_loss.data[0])


class MonitorMetricsXy(MonitorMetricsBase):
    '''Monitor metrics for classification and regression.
    Same as MonitorMetricsBase but we input a pair, X, y instead of data.

    For survival methods, see e.g. MonitorMetricsCox.

    Parameters:
        X: Numpy array with features.
        y: Numpy array with labels.
        monitor_funcs: Function, list, or dict of functions giving quiatities that should
            be monitored.
            The function takes argumess (df, preds) and should return a score.
        batch_size: Batch size used for calculating the scores.
        **kwargs: Can be passed to predict method.
    '''
    def __init__(self, X, y, monitor_funcs, per_epoch=1, batch_size=512, **kwargs):
        self.X, self.y = X, y
        self.kwargs = kwargs
        super().__init__(monitor_funcs, per_epoch, batch_size)

    def get_score_args(self):
        '''This function should create arguments to the monitor function.
        Typically it can return a tuple with (y_true, preds), to calculate e.g. auc.
        '''
        preds = self.model.predict(self.X, self.batch_size, **self.kwargs)
        return self.y, preds


class MonitorMetricsSklearn(MonitorMetricsXy):
    '''Class for monitoring metrics of from sklearn metrics

    Parameters:
        X: Numpy array with features.
        y: Numpy array with labels.
        monitor: Name for method in sklearn.metrics.
            E.g. 'log_loss', or pass sklearn.metrics.log_loss.
            For additional parameter, specify the function with a lambda statemetn.
                e.g. {'accuracy': lambda y, p: metrics.accuracy_score(y, p > 0.5)}
        batch_size: Batch size used for calculating the scores.
        **kwargs: Can be passed to predict method.
    '''
    def __init__(self, X, y, monitor, per_epoch=1, batch_size=512, **kwargs):
        if monitor.__class__ is str:
            monitor = {monitor: monitor}
        elif monitor.__class__ is list:
            monitor = {mon if mon.__class__ is str else str(i): mon
                       for i, mon in enumerate(monitor)}

        monitor = {name: getattr(metrics, mon) if mon.__class__ == str else mon
                   for name, mon in monitor.items()}
        super().__init__(X, y, monitor, per_epoch, batch_size)


class MonitorMetricsCox(MonitorMetricsBase):
    '''Class for monitoring metrics of survival during training progress.

    Parameters:
        df: Pandas dataframe with data used for monitoring.
        monitor: Quantity to be monitored. Dictionary with names and functions
            or list with names. Each function should take arguments df and g_preds
            and returning a scalar.
            Accepted strings are:
                {'cindex', 'mpll'}, or full names
                {'concordance_index', 'mean_partial_log_likelihood'}.
        batch_size: Batch size used for calculating the scores.
        **kwargs: Can be passed to predict method.
    '''
    def __init__(self, df, monitor, per_epoch=1, batch_size=512, **kwargs):
        self.df = df
        self.kwargs = kwargs
        if monitor.__class__ is str:
            monitor = {monitor: monitor}
        elif monitor.__class__ is list:
            monitor = {mon if mon.__class__ is str else str(i): mon
                       for i, mon in enumerate(monitor)}

        monitor = {name: self.__getattribute__('_'+mon) if mon.__class__ == str else mon
                   for name, mon in monitor.items()}
        super().__init__(monitor, per_epoch, batch_size)

    def get_score_args(self):
        '''This function should create arguments to the monitor function.
        Typically it can return a tuple with (y_true, preds), to calculate e.g. auc.
        '''
        g_preds = self.model.predict_g(self.df, self.batch_size, **self.kwargs)
        return self.df, g_preds

    def _cindex(self, df, g_preds):
        return self._concordance_index(df, g_preds)

    def _mpll(self, df, g_preds):
        return self._mean_partial_log_likelihood(df, g_preds)

    def _concordance_index(self, df, g_preds):
        return self.model.concordance_index(df, g_preds)

    def _mean_partial_log_likelihood(self, df, g_preds):
        return self.model.partial_log_likelihood(df, g_preds).mean()


class MonitorCoxLoss(MonitorMetricsBase):
    '''Class for monitoring validation loss in CoxNN during training progress.
    This works exactly like the loss in CoxNNTime, but it does not update the parameters of
    our model.

    It use the model's make_dataloader function to generate a dataloader for the validation
    set, and has it's own fit methods with eval=True in the net.

    Parameters:
        df: Pandas dataframe with data used for monitoring.
        n_control: Number of control samples.
        n_reps: Number of replications of the loss that are averaged.
            For smaller datasets it might give smoother validation curves.
        n_workers: Number of workers for preparing data.
        batch_size: Batch size used for calculating the scores.
        per_epoch: Calculated scores per epoch.
    '''
    def __init__(self, df, n_control, n_reps=1, n_workers=0, batch_size=1028, per_epoch=1):
        self.df = df
        self.n_control = n_control
        self.n_reps = n_reps
        self.n_workers = n_workers
        monitor = {'loss': self._loss}
        self._dataloader_exists = False
        super().__init__(monitor, per_epoch, batch_size)

    def get_score_args(self):
        '''This function should create arguments to the monitor function.
        Typically it can return a tuple with (y_true, preds), to calculate e.g. auc.
        '''
        return None, None

    def on_fit_start(self):
        if not self._dataloader_exists:
            self.df = self.df.sort_values(self.model.duration_col).reset_index()
            self.dataloader = self.make_dataloader()
        self._dataloader_exists = True

    def _loss(self, doesnt=None, matter=None):
        loss = self._run_dataloader()
        return loss

    def make_dataloader(self):
        X, time_fail, gr_alive = self._prepare_data()
        dataloader = self.model.make_dataloader(X, time_fail, gr_alive, self.n_control*self.n_reps,
                                                self.batch_size, self.n_workers)
        return dataloader

    def _prepare_data(self):
        model = self.model
        time_fail = self.df.loc[lambda x: x[model.event_col] == 1][model.duration_col]
        gr_alive = model._gr_alive(self.df, model.duration_col)
        X = self.df[model.x_columns].as_matrix().astype('float32')
        return X, time_fail, gr_alive

    def _run_dataloader(self):
        self.model.g.eval()
        loss = []
        for case, control in self.dataloader:
            if self.model.cuda:
                case, control = case.cuda(), control.cuda()
            case, control = Variable(case, volatile=True), Variable(control, volatile=True)
            g_case = self.model.g(case)
            g_control_all = [self.model.g(ctr) for ctr in control]
            # batch_loss = self.model.compute_loss(g_case, g_control)
            # loss.append(batch_loss.data[0])
            iters = np.arange(0, len(g_control_all)+self.n_control, self.n_control)
            g_control = [g_control_all[s:e]for s, e in zip(iters[:-1], iters[1:])]
            batch_loss = [self.model.compute_loss(g_case, gc).data[0] for gc in g_control]
            loss.append(np.mean(batch_loss))

        self.model.g.train()
        return np.mean(loss)


class MonitorCoxTimeLoss(MonitorCoxLoss):
    '''Class for monitoring validation loss in CoxNNTime.
    This works exactly like the loss in CoxNNTime, but it does not update the parameters of
    our model.

    It use the model's make_dataloader function to generate a dataloader for the validation
    set, and has it's own fit methods with eval=True in the net.

    Parameters:
        df: Pandas dataframe with data used for monitoring.
        n_control: Number of control samples.
        n_workers: Number of workers for preparing data.
        batch_size: Batch size used for calculating the scores.
        per_epoch: Calculated scores per epoch.
    '''
    def _prepare_data(self):
        model = self.model
        if self.df[model.duration_col].dtype != 'float32':
            as_32 = self.df[model.duration_col].astype('float32')
            self.df = self.df.assign(**{model.duration_col: as_32})

        return super()._prepare_data()

MonitorCoxNNLoss = MonitorCoxLoss  # Legacy
MonitorCoxNNTimeLoss  = MonitorCoxTimeLoss  # Legacy

class ClipGradNorm(Callback):
    '''Callback for clipping gradients.
    
    See torch.nn.utils.clip_grad_norm.

    Parameters:
        parameters: An iterable of Variables that will have gradients normalized.
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for infinity norm.

    '''
    def __init__(self, parameters, max_norm, norm_type=2):
        self.parameters = parameters
        self.max_norm = max_norm
        self.norm_type = norm_type

    def before_step(self):
        torch.nn.utils.clip_grad_norm(self.parameters, self.max_norm, self.norm_type)
        stop_signal = False
        return stop_signal

class LRScheduler(Callback):
    '''Wrapper for pytorch.optim.lr_scheduler objects.

    Parameters:
        scheduler: A pytorch.optim.lr_scheduler object.
        mm_obj: MonitorMetrics object, where first metric is used for early stopping.
            E.g. MonitorMetricsSurvival(df_val, 'cindex').
    '''
    def __init__(self, scheduler, mm_obj):
        self.scheduler = scheduler
        self.mm_obj = mm_obj

    def give_model(self, model):
        super().give_model(model)
        self.mm_obj.give_model(model)

    def on_fit_start(self):
        self.mm_obj.on_fit_start()

    def before_step(self):
        return self.mm_obj.before_step()

    def on_batch_end(self):
        self.mm_obj.on_batch_end()

    def on_epoch_end(self):
        self.mm_obj.on_epoch_end()
        score = self.mm_obj.scores[0][-1]
        self.scheduler.step(score)
        stop_signal = False
        return stop_signal
    