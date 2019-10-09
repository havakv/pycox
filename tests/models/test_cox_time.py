import pytest
from pycox.models import CoxTime
from pycox.models.cox_time import MLPVanillaCoxTime

from utils_model_testing import make_dataset, fit_model, assert_survs


@pytest.mark.parametrize('numpy', [True, False])
def test_cox_time_runs(numpy):
    data = make_dataset(False)
    data = data.apply(lambda x: x.float())
    if numpy:
        data = data.to_numpy()
    net = MLPVanillaCoxTime(data[0].shape[1], [4], False)
    model = CoxTime(net)
    fit_model(data, model)
    model.compute_baseline_hazards()
    assert_survs(data[0], model)
