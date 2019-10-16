import pytest
import torchtuples as tt
from pycox.models import CoxPH
from pycox.models.cox_time import MLPVanillaCoxTime

from utils_model_testing import make_dataset, fit_model, assert_survs


@pytest.mark.parametrize('numpy', [True, False])
def test_cox_cc_runs(numpy):
    data = make_dataset(False).apply(lambda x: x.float()).to_numpy()
    if not numpy:
        data = data.to_tensor()
    net = tt.practical.MLPVanilla(data[0].shape[1], [4], 1, False, output_bias=False)
    model = CoxPH(net)
    fit_model(data, model)
    model.compute_baseline_hazards()
    assert_survs(data[0], model)
