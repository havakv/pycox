
'''
File contains dataloaders.
'''
import numpy as np

import torch
import torch.utils.data as data

from .pytorch_dataloader import DataLoaderSlice


def sample_alive_from_dates(dates, gr_alive, n_control=1):
    '''Sample index from living at time given in dates.
    dates: np.array of times (or pd.Series).
    gr_alive: dict with gr_alive[time] = <array with index of alive in X matrix>.
    n_control: number of samples.

    !!!! This is now with replacement!!!!!

    '''
    lengths = np.array([gr_alive[x].shape[0] for x in dates])  # Can be moved outside
    idx = (np.random.uniform(size=(n_control, dates.size)) * lengths).astype('int')
    samp = np.empty((dates.size, n_control), dtype=int)
    samp.fill(np.nan)

    for it, time in enumerate(dates):
        samp[it, :] = gr_alive[time][idx[:, it]]
    return samp


class CoxPrepare(data.Dataset):
    '''Torch Dataset for preparing Cox case controll.

    Parameters:
    Xtr: np.array float32, with all training data.
    time_fail: pd.Series with index corresponding to failures in Xtr
        and values giving time of death (as int).
    gr_alive: dict with
            key: time of death
            val: index (Xtr) of alive at time 'key'.
    n_control: number of control samples.
    '''
    def __init__(self, Xtr, time_fail, gr_alive, n_control=1):
        self.Xtr = Xtr
        self.time_fail = time_fail
        self.gr_alive = gr_alive
        self.n_control = n_control

    def get_case_control(self, index):
        if not hasattr(index, '__iter__'):
            index = [index]
        fails = self.time_fail.iloc[index]

        x_case = self.Xtr[fails.index]
        control_idx = sample_alive_from_dates(fails.values, self.gr_alive, self.n_control)
        x_control = [self.Xtr[idx] for idx in control_idx.transpose()]
        return x_case, np.stack(x_control)

    def __getitem__(self, index):
        x_case, x_control = self.get_case_control(index)
        return torch.from_numpy(x_case), torch.from_numpy(x_control)

    def __len__(self):
        return self.time_fail.size


class CoxPrepareWithTime(CoxPrepare):
    '''Same as CoxPrepare, but time included as a covariate.

    Parameters:
    Xtr: np.array float32, with all training data.
    time_fail: pd.Series with index corresponding to failures in Xtr
        and values giving time of death (as int).
    gr_alive: dict with
            key: time of death
            val: index (Xtr) of alive at time 'key'.
    n_control: number of control samples.
    '''
    def _make_x_with_time(self, x, times, r, c):
        x_new = np.empty((r, c+1), dtype='float32')
        x_new[:, :c] = x
        x_new[:, c] = times
        return x_new

    def get_case_control(self, index):
        x_case, x_control = super().get_case_control(index)
        if not hasattr(index, '__iter__'):
            index = [index]
        fails = self.time_fail.iloc[index]
        r, c = len(index), self.Xtr.shape[1]
        x_case = self._make_x_with_time(x_case, fails.values, r, c)
        x_control = [self._make_x_with_time(x, fails.values, r, c)
                     for x in x_control]
        return x_case, np.stack(x_control)


class NumpyTensorDataset(data.Dataset):
    '''Turn np.array or list of np.array into a torch Dataset.
    X: numpy array or list of numpy arrays.
    '''
    def __init__(self, X):
        self.X = X
        self.single = False if X.__class__ is list else True
        if not self.single:
            assert len(set([len(x) for x in self.X])) == 1,\
                'All elements in X must have the same length.'

    def __getitem__(self, index):
        if self.single:
            return torch.from_numpy(self.X[index])
        return [torch.from_numpy(x[index]) for x in self.X]

    def __len__(self):
        return self.X.shape[0]

