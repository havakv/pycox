from pycox.datasets import from_deepsurv
from pycox.datasets import from_rdatasets
from pycox.datasets.kkbox import _DatasetKKBoxChurn


support = from_deepsurv._Support()
metabric = from_deepsurv._Metabric()
gbsg = from_deepsurv._Gbsg()
flchain = from_rdatasets._Flchain()
nwtco = from_rdatasets._Nwtco()
kkbox = _DatasetKKBoxChurn()