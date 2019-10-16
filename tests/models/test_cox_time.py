import pytest
import torchtuples as tt
from pycox.models import CoxTime
from pycox.models.cox_time import MLPVanillaCoxTime

from utils_model_testing import make_dataset, fit_model, assert_survs


@pytest.mark.parametrize('numpy', [True, False])
def test_cox_time_runs(numpy):
    input, target = make_dataset(False).apply(lambda x: x.float()).to_numpy()
    labtrans = CoxTime.label_transform()
    target = labtrans.fit_transform(*target)
    data = tt.tuplefy(input, target)
    if not numpy:
        data = data.to_tensor()
    net = MLPVanillaCoxTime(data[0].shape[1], [4], False)
    model = CoxTime(net)
    fit_model(data, model)
    model.compute_baseline_hazards()
    assert_survs(data[0], model)
