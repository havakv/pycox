'''
File containing a general fitter typically used for classification.
'''
# import numpy as np
import torch
# from torch.autograd import Variable
# import torch.optim as optim
from torch.utils.data import Dataset
# from ..dataloader import NumpyTensorDataset, DataLoaderSlice
# from ..callbacks import CallbacksList, TrainingLogger
# from .utils import to_cuda
from .base import BaseModel


class PrepareData(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, index):
        if not hasattr(index, '__iter__'):
            index = [index]
        return torch.from_numpy(self.X[index]), torch.from_numpy(self.y[index])

    def __len__(self):
        return self.y.shape[0]


class FitNet(BaseModel):
    def __init__(self, net, loss_func, optimizer=None, device=None):
        super().__init__(net, optimizer, device)
        self.loss_func = loss_func

    @staticmethod
    def make_dataloader(X, y, batch_size, num_workers):
        trainset = PrepareData(X, y)
        dataloader = DataLoaderSlice(trainset, batch_size=batch_size, shuffle=True,
                                     num_workers=num_workers)
        return dataloader

    def fit(self, X, y, batch_size=64, epochs=1, num_workers=0, callbacks=None, verbose=1):
        dataloader = self.make_dataloader(X, y, batch_size, num_workers)
        self._setup_train_info(dataloader, verbose, callbacks)

        self.callbacks.on_fit_start()
        for _ in range(epochs):
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                out = self.net(X)
                self.batch_loss = self.loss_func(out, y)
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

    def predict(self, X, batch_size=8224, return_numpy=True, eval_=True):
        return self._predict_func_numpy(self.net, batch_size, return_numpy, eval_)

# class FitNet(object):
#     def __init__(self, net, loss_func, optimizer=None, device=None):
#         self.net = net
#         self.loss_func = loss_func
#         self.optimizer = optimizer if optimizer else optim.SGD(self.net.parameters(), 0.01, 0.9)
#         # self.cuda = cuda
#         # if self.cuda is not False:
#         #     to_cuda(self.net, self.cuda)
#         if device is None:
#             device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         elif device.__class__ is str:
#             device = torch.device(device)
#         elif device.__class__ is int:
#             device = torch.device('cuda:{}'.format(device))
#         else:
#             if device.__class__ is not torch.device:
#                 raise ValueError('Argument `device` needs to be None, string, or torch.device object.')
#         self.device = device
#         self.net.to(self.device)

#         self.log = TrainingLogger()

#     @staticmethod
#     def make_dataset(Xtr, ytr, batch_size, num_workers):
#         trainset = PrepareData(Xtr, ytr)
#         dataloader = DataLoaderSlice(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
#         return dataloader

#     def fit(self, Xtr, ytr, batch_size=64, epochs=1, num_workers=0, callbacks=None, verbose=1):
#         self.log.verbose = verbose
#         dataloader = self.make_dataset(Xtr, ytr, batch_size, num_workers)

#         callbacks = callbacks if callbacks else []
#         self.callbacks = CallbacksList(callbacks + [self.log])
#         self.callbacks.give_model(self)
#         self.callbacks.on_fit_start()

#         for _ in range(epochs):
#             for Xtr, ytr in dataloader:
#                 # if self.cuda is not False:
#                 #     Xtr, ytr = to_cuda(Xtr, self.cuda), to_cuda(ytr, self.cuda)
#                 #     # Xtr, ytr = Xtr.cuda(), ytr.cuda()
#                 # Xtr, ytr = Variable(Xtr), Variable(ytr)
#                 Xtr, ytr = Xtr.to(self.device), ytr.to(self.device)
#                 out = self.net(Xtr)
#                 self.batch_loss = self.loss_func(out, ytr)
#                 self.optimizer.zero_grad()
#                 self.batch_loss.backward()
#                 self.optimizer.step()
#                 self.callbacks.on_batch_end()
#             stop_signal = self.callbacks.on_epoch_end()
#             if stop_signal:
#                 break
#         return self.log

#     def predict(self, X, batch_size=8224, return_numpy=True, eval_=True):
#         if eval_:
#             self.net.eval()
#         with torch.no_grad():
#             dataset = NumpyTensorDataset(X)
#             dataloader = DataLoaderSlice(dataset, batch_size)
#             preds = [self.net(x.to(self.device)) for x in iter(dataloader)]
#             preds = torch.cat(preds)
#         if eval_:
#             self.net.train()

#         if return_numpy:
#             return preds.numpy()
#         return preds
#     # def predict(self, X, batch_size=512, return_numpy=True, eval_=True):
#     #     if eval_:
#     #         self.net.eval()
#     #     if len(X) < batch_size:
#     #         if self.cuda is not False:
#     #             preds = [self.net(Variable(to_cuda(torch.from_numpy(X), self.cuda), volatile=True))]
#     #             # preds = [self.net(Variable(torch.from_numpy(X).cuda(), volatile=True))]
#     #         else:
#     #             preds = [self.net(Variable(torch.from_numpy(X), volatile=True))]
#     #     else:
#     #         dataset = NumpyTensorDataset(X)
#     #         dataloader = DataLoaderSlice(dataset, batch_size)
#     #         if self.cuda is not False:
#     #             preds = [self.net(Variable(to_cuda(x, self.cuda), volatile=True))
#     #                      for x in iter(dataloader)]
#     #         else:
#     #             preds = [self.net(Variable(x, volatile=True))
#     #                      for x in iter(dataloader)]
#     #     if eval_:
#     #         self.net.train()
#     #     if return_numpy:
#     #         if self.cuda is not False:
#     #             preds = [pred.data.cpu().numpy() for pred in preds]
#     #         else:
#     #             preds = [pred.data.numpy() for pred in preds]
#     #         return np.concatenate(preds)
#     #     return preds

#     def save_model_weights(self, path, **kwargs):
#         '''Save the model weights.

#         Parameters:
#             path: The filepath of the model.
#             **kwargs: Arguments passed to torch.save method.
#         '''
#         # if self.cuda is not False:
#         #     # If we don't do this, torch will move g to default gpu.
#         #     # Remove this when bug is fixed...
#         #     self.net.cpu()
#         #     torch.save(self.net.state_dict(), path, **kwargs)
#         #     to_cuda(self.net, self.cuda)
#         # else:
#         #     torch.save(self.net.state_dict(), path, **kwargs)
#         return torch.save(self.net.state_dict(), path, **kwargs)

#     def load_model_weights(self, path, **kwargs):
#         '''Load model weights.

#         Parameters:
#             path: The filepath of the model.
#             **kwargs: Arguments passed to torch.load method.
#         '''
#         self.net.load_state_dict(torch.load(path, **kwargs))
