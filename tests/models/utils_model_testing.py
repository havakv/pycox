import pandas as pd
import torch
import torchtuples as tt

def make_dataset(numpy):
    n_events = 2
    n_frac = 4
    m = 10
    n = m * n_frac * n_events
    p = 5
    input = torch.randn((n, p))
    durations = torch.arange(m).repeat(int(n / m))
    events = torch.arange(n_events).repeat(int(n / n_events)).float()
    target = (durations, events)
    data  = tt.tuplefy(input, target)
    if numpy:
        data = data.to_numpy()
    return data

def fit_model(data, model):
    model.fit(*data, epochs=1, verbose=False, val_data=data)
    return model

def assert_survs(input, model):
    preds = model.predict_surv(input)
    assert type(preds) is type(input)
    surv_df = model.predict_surv_df(input)
    assert type(surv_df) is pd.DataFrame
    np_input = tt.tuplefy(input).to_numpy()[0]
    torch_input = tt.tuplefy(input).to_tensor()[0]
    np_preds = model.predict_surv(np_input)
    torch_preds = model.predict_surv(torch_input)
    assert (np_preds == torch_preds.numpy()).all()
