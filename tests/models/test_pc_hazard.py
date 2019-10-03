import pytest
import torch
import numpy as np
from pycox.models import PCHazard


def _make_dataset(n, m):
    np.random.seed(0)
    x = np.random.normal(0, 1, (n, 4)).astype('float32')
    duration_index = np.arange(m+1)
    durations = np.repeat(duration_index, np.ceil(n / m))[:n]
    events = np.random.uniform(0, 1, n).round().astype('float32')
    fracs = np.random.uniform(0, 1, n).astype('float32')
    return x, (durations, events, fracs), duration_index

@pytest.mark.parametrize('m', [5, 10])
@pytest.mark.parametrize('n_mul', [2, 3])
@pytest.mark.parametrize('mp', [1, 2, -1])
def test_wrong_net_output(m, n_mul, mp):
    n = m * n_mul
    inp, tar, dur_index = _make_dataset(n, m)
    net = torch.nn.Linear(inp.shape[1], m+1)
    with pytest.raises(ValueError):
        model = PCHazard(net, duration_index=dur_index)

    model = PCHazard(net)
    with pytest.raises(ValueError):
        model.fit(inp, tar)

    model.duration_index = dur_index
    with pytest.raises(ValueError):
        model.predict_surv_df(inp)

    model.duration_index = dur_index
    dl = model.make_dataloader((inp, tar), 5, True)
    with pytest.raises(ValueError):
        model.fit_dataloader(dl)

@pytest.mark.parametrize('m', [5, 10])
@pytest.mark.parametrize('n_mul', [2, 3])
def test_right_net_output(m, n_mul):
    n = m * n_mul
    inp, tar, dur_index = _make_dataset(n, m)
    net = torch.nn.Linear(inp.shape[1], m)
    model = PCHazard(net)
    model = PCHazard(net, duration_index=dur_index)
    model.fit(inp, tar, verbose=False)
    model.predict_surv_df(inp)
    dl = model.make_dataloader((inp, tar), 5, True)
    model.fit_dataloader(dl)
    assert True
