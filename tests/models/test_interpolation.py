import pytest
import torch
from pycox import models
from pycox.models.interpolation import InterpolateDiscrete


class MockPMF(models.PMF):
    def __init__(self, duration_index=None):
        self.duration_index = duration_index
        
    def predict(self, input, *args, **kwargs):
        return input

class MockLogisticHazard(models.LogisticHazard):
    def __init__(self, duration_index=None):
        self.duration_index = duration_index
        
    def predict(self, input, *args, **kwargs):
        return input


@pytest.mark.parametrize('m', [2, 5, 10])
@pytest.mark.parametrize('sub', [2, 5])
def test_pmf_cdi_equals_base(m, sub):
    torch.manual_seed(12345)
    n = 20
    idx = torch.randn(m).abs().sort()[0].numpy()
    input = torch.randn(n, m)
    model = MockPMF(idx)
    surv_pmf = model.interpolate(sub).predict_surv_df(input)
    surv_base = InterpolateDiscrete(model, duration_index=idx, sub=sub).predict_surv_df(input)
    assert (surv_pmf.index == surv_base.index).all()
    assert (surv_pmf - surv_base).abs().max().max() < 1e-7


@pytest.mark.parametrize('m', [2, 5, 10])
@pytest.mark.parametrize('sub', [2, 5])
def test_base_values_at_knots(m, sub):
    torch.manual_seed(12345)
    n = 20
    idx = torch.randn(m).abs().sort()[0].numpy()
    input = torch.randn(n, m)
    model = MockPMF(idx)
    surv_cdi = InterpolateDiscrete(model, duration_index=idx, sub=sub).predict_surv_df(input)
    surv = model.predict_surv_df(input)
    diff = (surv - surv_cdi).dropna()
    assert diff.shape == surv.shape
    assert (diff == 0).all().all()


@pytest.mark.parametrize('m', [2, 5, 10])
@pytest.mark.parametrize('sub', [2, 5])
def test_logistic_hazard_values_at_knots(m, sub):
    torch.manual_seed(12345)
    n = 20
    idx = torch.randn(m).abs().sort()[0].numpy()
    input = torch.randn(n, m)
    model = MockLogisticHazard(idx)
    surv = model.predict_surv_df(input)
    surv_cdi = model.interpolate(sub, 'const_pdf').predict_surv_df(input)
    diff = (surv - surv_cdi).dropna()
    assert diff.shape == surv.shape
    assert (diff == 0).all().all()
    surv_chi = model.interpolate(sub, 'const_hazard').predict_surv_df(input)
    diff = (surv - surv_chi).dropna()
    assert diff.shape == surv.shape
    assert (diff.index == surv.index).all()
    assert diff.max().max() < 1e-6
