
import numpy as np
import pandas as pd
import torch
from pyth import Model, tuplefy, make_dataloader, TupleTree
from pyth.data import DatasetTuple


def sample_alive_from_dates(dates, at_risk_dict, n_control=1):
    '''Sample index from living at time given in dates.
    dates: np.array of times (or pd.Series).
    gr_alive: dict with gr_alive[time] = <array with index of alive in X matrix>.
    n_control: number of samples.

    !!!! This is now with replacement!!!!!

    '''
    lengths = np.array([at_risk_dict[x].shape[0] for x in dates])  # Can be moved outside
    idx = (np.random.uniform(size=(n_control, dates.size)) * lengths).astype('int')
    samp = np.empty((dates.size, n_control), dtype=int)
    samp.fill(np.nan)

    for it, time in enumerate(dates):
        samp[it, :] = at_risk_dict[time][idx[:, it]]
    return samp

def make_at_risk_dict(durations):
    '''Create dict(duration: indices) from sorted df.

    Parameters:
        df: A Pandas dataframe with covariates, sorted by duration_col.
        duration_col: Column holding the durations.

    Returns:
        A dict mapping durations to indices (row number, not index in data frame).
        For each time => index of all individual alive.
    '''
    assert type(durations) is np.ndarray, 'Need durations to be a numpy array'
    durations = pd.Series(durations)
    assert durations.is_monotonic_increasing, 'Requires durations to be monotonic'
    allidx = durations.index.values
    keys = durations.drop_duplicates(keep='first')
    at_risk_dict = dict()
    for ix, t in keys.iteritems():
        at_risk_dict[t] = allidx[ix:]
    return at_risk_dict

class DatasetDurationSorted(DatasetTuple):
    def __getitem__(self, index):
        batch = super().__getitem__(index)
        input, (duration, event) = batch
        idx_sort = duration.sort(descending=True)[1]
        event = event.float()
        batch = tuplefy(input, event).iloc[idx_sort]
        return batch

class CoxCCPrepare(torch.utils.data.Dataset):
    def __init__(self, input, durations, events, n_control=1):
        df_train_target = pd.DataFrame(dict(duration=durations, event=events))
        self.durations = df_train_target.loc[lambda x: x['event'] == 1]['duration']
        self.at_risk_dict = make_at_risk_dict(durations)

        self.input = tuplefy(input)
        assert type(self.durations) is pd.Series
        self.n_control = n_control

    def __getitem__(self, index):
        if not hasattr(index, '__iter__'):
            index = [index]
        fails = self.durations.iloc[index]
        x_case = self.input.iloc[fails.index]
        control_idx = sample_alive_from_dates(fails.values, self.at_risk_dict, self.n_control)
        x_control = TupleTree(self.input.iloc[idx] for idx in control_idx.transpose())
        return tuplefy(x_case, x_control).to_tensor(), None

    def __len__(self):
        return len(self.durations)


class CoxCCPrepare(torch.utils.data.Dataset):
    def __init__(self, input, durations, events, n_control=1):
        df_train_target = pd.DataFrame(dict(duration=durations, event=events))
        self.durations = df_train_target.loc[lambda x: x['event'] == 1]['duration']
        self.at_risk_dict = make_at_risk_dict(durations)

        self.input = tuplefy(input)
        assert type(self.durations) is pd.Series
        self.n_control = n_control

    def __getitem__(self, index):
        if not hasattr(index, '__iter__'):
            index = [index]
        fails = self.durations.iloc[index]
        x_case = self.input.iloc[fails.index]
        control_idx = sample_alive_from_dates(fails.values, self.at_risk_dict, self.n_control)
        x_control = TupleTree(self.input.iloc[idx] for idx in control_idx.transpose())
        return tuplefy(x_case, x_control).to_tensor(), None

    def __len__(self):
        return len(self.durations)


class CoxTimePrepare(CoxCCPrepare):
    def __init__(self, input, durations, events, n_control=1):
        super().__init__(input, durations, events, n_control)
        self.durations_tensor = tuplefy(self.durations.values.reshape(-1, 1)).to_tensor()

    def __getitem__(self, index):
        if not hasattr(index, '__iter__'):
            index = [index]
        durations = self.durations_tensor.iloc[index]
        (case, control), _ = super().__getitem__(index)
        case = case + durations
        control = control.apply_nrec(lambda x: x + durations)
        return tuplefy(case, control), None