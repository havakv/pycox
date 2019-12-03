
import numpy as np
import pandas as pd
import numba
import torch
import torchtuples as tt


def sample_alive_from_dates(dates, at_risk_dict, n_control=1):
    '''Sample index from living at time given in dates.
    dates: np.array of times (or pd.Series).
    at_risk_dict: dict with at_risk_dict[time] = <array with index of alive in X matrix>.
    n_control: number of samples.
    '''
    lengths = np.array([at_risk_dict[x].shape[0] for x in dates])  # Can be moved outside
    idx = (np.random.uniform(size=(n_control, dates.size)) * lengths).astype('int')
    samp = np.empty((dates.size, n_control), dtype=int)
    samp.fill(np.nan)

    for it, time in enumerate(dates):
        samp[it, :] = at_risk_dict[time][idx[:, it]]
    return samp

def make_at_risk_dict(durations):
    """Create dict(duration: indices) from sorted df.
    A dict mapping durations to indices.
    For each time => index of all individual alive.
    
    Arguments:
        durations {np.arrary} -- durations.
    """
    assert type(durations) is np.ndarray, 'Need durations to be a numpy array'
    durations = pd.Series(durations)
    assert durations.is_monotonic_increasing, 'Requires durations to be monotonic'
    allidx = durations.index.values
    keys = durations.drop_duplicates(keep='first')
    at_risk_dict = dict()
    for ix, t in keys.iteritems():
        at_risk_dict[t] = allidx[ix:]
    return at_risk_dict


class DurationSortedDataset(tt.data.DatasetTuple):
    """We assume the dataset contrain `(input, durations, events)`, and 
    sort the batch based on descending `durations`.

    See `torchtuples.data.DatasetTuple`.
    """
    def __getitem__(self, index):
        batch = super().__getitem__(index)
        input, (duration, event) = batch
        idx_sort = duration.sort(descending=True)[1]
        event = event.float()
        batch = tt.tuplefy(input, event).iloc[idx_sort]
        return batch


class CoxCCDataset(torch.utils.data.Dataset):
    def __init__(self, input, durations, events, n_control=1):
        df_train_target = pd.DataFrame(dict(duration=durations, event=events))
        self.durations = df_train_target.loc[lambda x: x['event'] == 1]['duration']
        self.at_risk_dict = make_at_risk_dict(durations)

        self.input = tt.tuplefy(input)
        assert type(self.durations) is pd.Series
        self.n_control = n_control

    def __getitem__(self, index):
        if (not hasattr(index, '__iter__')) and (type(index) is not slice):
            index = [index]
        fails = self.durations.iloc[index]
        x_case = self.input.iloc[fails.index]
        control_idx = sample_alive_from_dates(fails.values, self.at_risk_dict, self.n_control)
        x_control = tt.TupleTree(self.input.iloc[idx] for idx in control_idx.transpose())
        return tt.tuplefy(x_case, x_control).to_tensor()

    def __len__(self):
        return len(self.durations)


class CoxTimeDataset(CoxCCDataset):
    def __init__(self, input, durations, events, n_control=1):
        super().__init__(input, durations, events, n_control)
        self.durations_tensor = tt.tuplefy(self.durations.values.reshape(-1, 1)).to_tensor()

    def __getitem__(self, index):
        if not hasattr(index, '__iter__'):
            index = [index]
        durations = self.durations_tensor.iloc[index]
        case, control = super().__getitem__(index)
        case = case + durations
        control = control.apply_nrec(lambda x: x + durations)
        return tt.tuplefy(case, control)

@numba.njit
def _pair_rank_mat(mat, idx_durations, events, dtype='float32'):
    n = len(idx_durations)
    for i in range(n):
        dur_i = idx_durations[i]
        ev_i = events[i]
        if ev_i == 0:
            continue
        for j in range(n):
            dur_j = idx_durations[j]
            ev_j = events[j]
            if (dur_i < dur_j) or ((dur_i == dur_j) and (ev_j == 0)):
                mat[i, j] = 1
    return mat

def pair_rank_mat(idx_durations, events, dtype='float32'):
    """Indicator matrix R with R_ij = 1{T_i < T_j and D_i = 1}.
    So it takes value 1 if we observe that i has an event before j and zero otherwise.
    
    Arguments:
        idx_durations {np.array} -- Array with durations.
        events {np.array} -- Array with event indicators.
    
    Keyword Arguments:
        dtype {str} -- dtype of array (default: {'float32'})
    
    Returns:
        np.array -- n x n matrix indicating if i has an observerd event before j.
    """
    idx_durations = idx_durations.reshape(-1)
    events = events.reshape(-1)
    n = len(idx_durations)
    mat = np.zeros((n, n), dtype=dtype)
    mat = _pair_rank_mat(mat, idx_durations, events, dtype)
    return mat


class DeepHitDataset(tt.data.DatasetTuple):
    def __getitem__(self, index):
        input, target =  super().__getitem__(index)
        target = target.to_numpy()
        rank_mat = pair_rank_mat(*target)
        target = tt.tuplefy(*target, rank_mat).to_tensor()
        return tt.tuplefy(input, target)
