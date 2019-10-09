import pytest
from pycox.models import MTLR
import torchtuples as tt

from utils_model_testing import make_dataset, fit_model, assert_survs


@pytest.mark.parametrize('numpy', [True, False])
@pytest.mark.parametrize('num_durations', [2, 5])
def test_mtlr_runs(numpy, num_durations):
    data = make_dataset(True)
    input, target = data
    labtrans = MTLR.label_transform(num_durations)
    target = labtrans.fit_transform(*target)
    data = tt.tuplefy(input, target)
    if not numpy:
        data = data.to_tensor()
    net = tt.practical.MLPVanilla(input.shape[1], [4], num_durations)
    model = MTLR(net)
    fit_model(data, model)
    assert_survs(input, model)
    model.duration_index = labtrans.cuts
    assert_survs(input, model)
    cdi = model.interpolate(3, 'const_pdf')
    assert_survs(input, cdi)
