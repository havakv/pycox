'''Base models.
'''
from collections import OrderedDict
import torch
from torch import optim
# from ..callbacks.callbacks import CallbacksList, TrainingLogger
from ..callbacks import callbacks as cb
from ..dataloader import NumpyTensorDataset, DataLoaderSlice

class BaseModel(object):
    '''Abstract base model.

    Parameters:
        net: Pytorch Module.
        optimizer: Torch optimizer. If None SGD(lr=0.1, momentum=0.9)
        device: Which device to compute on.
            Preferrably pass a torch.device object.
            If `None`: use default gpu if avaiable, else use cpu.
            If `int`: used that gpu: torch.device('cuda:<device>').
            If `string`: string is passed to torch.device(`string`).
    '''
    def __init__(self, net, optimizer=None, device=None):
        self.net = net
        # self.loss_func = loss_func
        self.optimizer = optimizer if optimizer else optim.SGD(self.net.parameters(), 0.01, 0.9)

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif device.__class__ is str:
            device = torch.device(device)
        elif device.__class__ is int:
            device = torch.device('cuda:{}'.format(device))
        else:
            if device.__class__ is not torch.device:
                raise ValueError('Argument `device` needs to be None, string, or torch.device object.')
        self.device = device
        self.net.to(self.device)

        self.train_loss = cb.MonitorTrainLoss()
        self.log = cb.TrainingLogger()
        self.log.monitors = OrderedDict(loss=self.train_loss)
    
    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
    
    def make_dataloader(self):
        raise NotImplementedError
    
    def _setup_train_info(self, dataloader, verbose, callbacks=None):
        self.fit_info = {'batches_per_epoch': len(dataloader)}

        self.log.verbose = verbose
        if callbacks is None:
            callbacks = []
        self.callbacks = cb.CallbacksList([self.train_loss] + callbacks + [self.log])
        self.callbacks.give_model(self)

    def _predict_func_numpy(self, func, X, batch_size=8224, return_numpy=True, eval_=True):
        '''Get func(X) for a numpy array X.

        Parameters:
            func: Pytorch module.
            X: Numpy matrix with with covariates.
            batch_size: Batch size.
            return_numpy: If False, a torch tensor is returned.
            eval_: If true, set `fun` in eval mode for prediction
                and back to train mode after that (only affects dropout and batchnorm).
                If False, leaves `fun` modes as they are.
        '''
        if eval_:
            func.eval()
        with torch.no_grad():
            dataset = NumpyTensorDataset(X)
            dataloader = DataLoaderSlice(dataset, batch_size)
            preds = [func(x.to(self.device)) for x in iter(dataloader)]
            preds = torch.cat(preds)
        if eval_:
            func.train()

        if return_numpy:
            return preds.numpy()
        return preds

    def save_model_weights(self, path, **kwargs):
        '''Save the model weights.

        Parameters:
            path: The filepath of the model.
            **kwargs: Arguments passed to torch.save method.
        '''
        return torch.save(self.net.state_dict(), path, **kwargs)

    def load_model_weights(self, path, **kwargs):
        '''Load model weights.

        Parameters:
            path: The filepath of the model.
            **kwargs: Arguments passed to torch.load method.
        '''
        self.net.load_state_dict(torch.load(path, **kwargs))
