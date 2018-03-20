'''
File containing a general fitter typically used for classification.
'''
import numpy as np
import torch
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import Dataset
from .dataloader import NumpyTensorDataset, DataLoaderSlice
from .callbacks import CallbacksList, TrainingLogger
from .utils import to_cuda


class PrepareData(Dataset):
    def __init__(self, Xtr, ytr):
        self.Xtr = Xtr
        self.ytr = ytr

    def __getitem__(self, index):
        if not hasattr(index, '__iter__'):
            index = [index]
        return torch.from_numpy(self.Xtr[index]), torch.from_numpy(self.ytr[index])

    def __len__(self):
        return self.ytr.shape[0]


class FitNet(object):
    def __init__(self, net, loss_func, optimizer=None, cuda=False):
        self.net = net
        self.loss_func = loss_func
        self.optimizer = optimizer if optimizer else optim.SGD(self.net.parameters(), 0.01, 0.9)
        self.cuda = cuda
        if self.cuda is not False:
            to_cuda(self.net, self.cuda)

        self.log = TrainingLogger()

    @staticmethod
    def make_dataset(Xtr, ytr, batch_size, num_workers):
        trainset = PrepareData(Xtr, ytr)
        dataloader = DataLoaderSlice(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        return dataloader

    def fit(self, Xtr, ytr, batch_size=64, epochs=1, num_workers=0, callbacks=None, verbose=1):
        self.log.verbose = verbose
        dataloader = self.make_dataset(Xtr, ytr, batch_size, num_workers)

        callbacks = callbacks if callbacks else []
        self.callbacks = CallbacksList(callbacks + [self.log])
        self.callbacks.give_model(self)
        self.callbacks.on_fit_start()

        for _ in range(epochs):
            for Xtr, ytr in dataloader:
                if self.cuda is not False:
                    Xtr, ytr = to_cuda(Xtr, self.cuda), to_cuda(ytr, self.cuda)
                    # Xtr, ytr = Xtr.cuda(), ytr.cuda()
                Xtr, ytr = Variable(Xtr), Variable(ytr)
                out = self.net(Xtr)
                self.batch_loss = self.loss_func(out, ytr)
                self.optimizer.zero_grad()
                self.batch_loss.backward()
                self.optimizer.step()
                self.callbacks.on_batch_end()
            stop_signal = self.callbacks.on_epoch_end()
            if stop_signal:
                break
        return self.log

    def predict(self, X, batch_size=512, return_numpy=True, eval_=True):
        if eval_:
            self.net.eval()
        if len(X) < batch_size:
            if self.cuda is not False:
                preds = [self.net(Variable(to_cuda(torch.from_numpy(X), self.cuda), volatile=True))]
                # preds = [self.net(Variable(torch.from_numpy(X).cuda(), volatile=True))]
            else:
                preds = [self.net(Variable(torch.from_numpy(X), volatile=True))]
        else:
            dataset = NumpyTensorDataset(X)
            dataloader = DataLoaderSlice(dataset, batch_size)
            if self.cuda is not False:
                preds = [self.net(Variable(to_cuda(x, self.cuda), volatile=True))
                         for x in iter(dataloader)]
            else:
                preds = [self.net(Variable(x, volatile=True))
                         for x in iter(dataloader)]
        if eval_:
            self.net.train()
        if return_numpy:
            if self.cuda is not False:
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
        torch.save(self.net.state_dict(), path, **kwargs)

    def load_model_weights(self, path, **kwargs):
        '''Load model weights.

        Parameters:
            path: The filepath of the model.
            **kwargs: Arguments passed to torch.load method.
        '''
        self.net.load_state_dict(torch.load(path, **kwargs))
