from pycox.datasets import from_deepsurv
from pycox.datasets import from_rdatasets
from pycox.datasets import from_kkbox
from pycox.datasets import from_simulations


support = from_deepsurv._Support()
metabric = from_deepsurv._Metabric()
gbsg = from_deepsurv._Gbsg()
flchain = from_rdatasets._Flchain()
nwtco = from_rdatasets._Nwtco()
kkbox_v1 = from_kkbox._DatasetKKBoxChurn()
kkbox = from_kkbox._DatasetKKBoxAdmin()
sac3 = from_simulations._SAC3()
rr_nl_nhp = from_simulations._RRNLNPH()
sac_admin5 = from_simulations._SACAdmin5()
