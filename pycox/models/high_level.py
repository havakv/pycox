'''Hight level models that are easy to use.
'''

import warnings
import torch
from torch import nn, optim
from .cox import CoxPH, CoxTime
from .torch_models import ReluNet
from ..callbacks import callbacks as cb

class CoxPHLinear(CoxPH):
    '''This class implements Cox's proportional hazards model:
    h(t|x) = h_0(t)*exp(g(x)), where g(x) = beta^T x.

    Parameters:
        input_size: Size of x, i.e. number of covariates.
        set_optim_func: Function for setting pytorch optimizer.
            If None optimizer is set to Adam with default parameters.
            Function should take one argument (pytorch model) and return the optimizer.
            See CoxPHLinear.set_optim_default as an example.
        device: Which device to compute on.
            Preferrably pass a torch.device object.
            If `None`: use default gpu if avaiable, else use cpu.
            If `string`: string is passed to torch.device(`string`).
    '''
    def __init__(self, input_size, set_optim_func=None, device=None):
        self.input_size = input_size
        g_model = self._make_g_model(self.input_size)
        self.set_optim_func = set_optim_func
        if self.set_optim_func is None:
            self.set_optim_func = self.set_optim_default
        optimizer = self.set_optim_func(g_model)
        super().__init__(g_model, optimizer, device)

    @staticmethod
    def set_optim_default(g_model):
        return optim.Adam(g_model.parameters())

    def _make_g_model(self, input_size):
        return nn.Sequential(nn.Linear(input_size, 1, bias=False))

    def fit(self, df, duration_col, event_col=None, n_control=1, batch_size=64, epochs=500,
            n_workers=0, verbose=1, strata=None, callbacks=None, compute_hazards=True,
            early_stop_train_patience=None):
        '''Fit the Cox Propertional Hazards model to a dataset. Tied survival times
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
                catagorical covariate does not obey the proportional hazards assumption. This
                is used similar to the `strata` expression in R.
                See http://courses.washington.edu/b515/l17.pdf.
            callbacks: List of callbacks.
            compute_hazards: If we should compute hazards when training has finished.
            early_stop_train_patience: Early stopping if train loss does not improve
                for `early_stop_train_patience` epochs.
                If `None`, this as no effect.

        # Returns:
        #     self, with additional properties: hazards_
        '''
        if callbacks is None:
            callbacks = []
        if early_stop_train_patience is not None:
            if early_stop_train_patience.__class__ is not int:
                raise ValueError('`early_stop_patience` needs to be `int`')
            # callbacks.append(EarlyStoppingTrainLoss())
            # train_loss = cb.MonitorTrainLoss()
            es = cb.EarlyStopping(self.train_loss, minimize=True,
                                  patience=early_stop_train_patience)
            callbacks.append(es)
        return super().fit(df, duration_col, event_col, n_control, batch_size, epochs,
                           n_workers, verbose, strata, callbacks, compute_hazards)


class _AbstractCoxReluNet(object):
    '''This class does not make sense alone, and need to be merged with a Cox class.

    Parameters:
        input_size: Input size.
        n_layers: Number of layers.
        n_nodes: Size of each layer.
        dropout: Dropout rate. If `False`, no dropout.
        batch_norm: If use of batch norm.
        optimizer_func: Set optimizer.
            E.g. optimizer_func = lambda x: optim.Adam(x, lr=0.1)
            If `None`: use defult Adam optimizer.
        device: Which device to compute on.
            Preferrably pass a torch.device object.
            If `None`: use default gpu if avaiable, else use cpu.
            If `int`: used that gpu: torch.device('cuda:<device>').
            If `string`: string is passed to torch.device(`string`).
    '''
    def __init__(self, input_size, n_layers, n_nodes, dropout=False, batch_norm=True,
                 optimizer_func=None, device=None):
        g = ReluNet(input_size, n_layers, n_nodes, dropout=False, batch_norm=True)
        if optimizer_func is not None:
            optimizer = optimizer_func(g.parameters())
        else:
            optimizer = torch.optim.Adam(g.parameters())
        super().__init__(g, optimizer, device)
    
    def fit(self, df_train, duration_col, event_col=None, df_val=None, batch_size=64, epochs=10,
            num_workers=0, verbose=True, compute_hazards=True, n_control=1,
            early_stop_patience=None, model_path=None):
        '''
        Parameters:
            df_train: A Pandas dataframe with necessary columns `duration_col` and
                `event_col`, plus other covariates. `duration_col` refers to
                the lifetimes of the subjects. `event_col` refers to whether
                the 'death' events was observed: 1 if observed, 0 else (censored).
            duration_col: The column in dataframe that contains the subjects'
                lifetimes.
            event_col: The column in dataframe that contains the subjects' death
                observation. If left as None, assume all individuals are non-censored.
            df_val: Same as `df_train` but validation data.
            batch_size: Batch size.
            epochs: Number of epochs.
            num_workers: Number of workers for preparing data.
            verbose: If we should print progress.
            compute_hazards: If we should compute hazards when training has finished.
            n_control: Number of control samples.
            early_stop_patience: Stop based when validation loss has not improved
                in `early_stop_patience` iterations.
            model_path: If use of early stopping, save best model to `model_path`.

        # Returns:
        #     self, with additional properties: hazards_
        '''
        
        warnings.warn('Need to fix max time for baseline!!!')

        callbacks = []
        first_fit = not hasattr(self, '_fit_params')
        if not first_fit:
            # callbacks = self.callbacks.callbacks[:-1]
            if self._fit_params['n_control'] != n_control:
                raise ValueError('Need n_control to be fixed to get comparable results.')
            if self._fit_params['df_val'] is not df_val:
                raise ValueError('Need `df_val` not to change.') 

        if df_val is not None:
            if first_fit:
                self.val_loss = cb.MonitorCoxTimeLoss(df_val, n_control, n_reps=5, num_workers=num_workers,
                                                      batch_size=batch_size)
                mon = self.log.monitors.copy()
                mon.update({'val': self.val_loss})
                self.log.monitors = mon

            callbacks.append(self.val_loss)

            # if verbose:
            #     verbose = {'val': self.val_loss}
                # if verbose.__class__ is not dict:
                #     verbose = {}
                # verbose.update(val_verb)
        if early_stop_patience is None:
            early_stop_patience = 9223372036854775807
    
        # if early_stop_patience is not None:
        if early_stop_patience.__class__ is not int:
            raise ValueError('`early_stop_patience` needs to be `int`')
        if df_val is None:
            raise ValueError('Need `df_val` to use `early_stop_patience`')
        es = cb.EarlyStopping(self.val_loss, minimize=True, patience=early_stop_patience,
                                model_file_path=model_path)
        callbacks.append(es)
        
        self._fit_params = {'batch_size': batch_size, 'num_workers': num_workers,
                            'verbose': verbose, 'n_control': n_control,
                            'early_stop_patience': early_stop_patience,
                            'df_val': df_val,}
        strata = None
        # super refers to a CoxPH or CoxTime model.
        return super().fit(df_train, duration_col, event_col, n_control, batch_size, epochs,
            num_workers, verbose, strata, callbacks, compute_hazards)

class CoxPHReluNet(_AbstractCoxReluNet, CoxPH):
    pass


class CoxTimeReluNet(_AbstractCoxReluNet, CoxTime):
    def fit(self, df_train, duration_col, event_col=None, df_val=None, batch_size=64, epochs=10,
            num_workers=0, verbose=True, compute_hazards=False, n_control=1,
            early_stop_patience=None, model_path=None):
        if (df_train.shape[1] - 1) != self.g.input_size:
            raise ValueError('''
            Input size does not fit network input.
            Remember that the network need to have 1 input for time.
            ''')
        return super().fit(df_train, duration_col, event_col, df_val, batch_size, epochs,
                           num_workers, verbose, compute_hazards, n_control,
                           early_stop_patience)
    # def __init__(self, input_size, n_layers, n_nodes, dropout=False, batch_norm=True,
    #              optimizer=None, device=None):
    #     g = ReluNet(input_size, n_layers, n_nodes, dropout=False, batch_norm=True)
    #     super().__init__(g, optimizer, device)
    
    # def fit(self, df_train, duration_col, event_col=None, df_val=None, batch_size=64, epochs=10,
    #         n_workers=0, verbose=True, callbacks=None, compute_hazards=False, n_control=1):
    #     '''
    #     TODO:
    #         Warn if input_size does not fit df! Give better warning than pytorch!!!!!!!!
    #     '''