from collections import defaultdict
import requests
import h5py
import pandas as pd
from pycox.datasets._dataset_loader import _DatasetLoader


class _DatasetDeepSurv(_DatasetLoader):
    _dataset_url = "https://raw.githubusercontent.com/jaredleekatzman/DeepSurv/master/experiments/data/"
    _datasets = {
        'support': "support/support_train_test.h5",
        'metabric': "metabric/metabric_IHC4_clinical_train_test.h5",
        'gbsg': "gbsg/gbsg_cancer_train_test.h5",
    }
    col_duration = 'duration'
    col_event = 'event'
    def _download(self):
        url = self._dataset_url + self._datasets[self.name]
        path = self.path.parent / f"{self.name}.h5"
        with requests.Session() as s:
            r = s.get(url)
            with open(path, 'wb') as f:
                f.write(r.content)

        data = defaultdict(dict)
        with h5py.File(path) as f:
            for ds in f: 
                for array in f[ds]:
                    data[ds][array] = f[ds][array][:]
        
        path.unlink()
        train = _make_df(data['train'])
        test = _make_df(data['test'])
        df = pd.concat([train, test]).reset_index(drop=True)
        df.to_feather(self.path)


def _make_df(data):
    x = data['x']
    t = data['t']
    d = data['e']

    colnames = ['x'+str(i) for i in range(x.shape[1])]
    df = (pd.DataFrame(x, columns=colnames)
          .assign(duration=t)
          .assign(event=d))
    return df


class _Support(_DatasetDeepSurv):
    """Study to Understand Prognoses Preferences Outcomes and Risks of Treatment (SUPPORT).

    A study of survival for seriously ill hospitalized adults.

    This is the processed data set used in the DeepSurv paper (Katzman et al. 2018), and details
    can be found at https://doi.org/10.1186/s12874-018-0482-1

    See https://github.com/jaredleekatzman/DeepSurv/tree/master/experiments/data
    for original data.
    """
    name = 'support'


class _Metabric(_DatasetDeepSurv):
    """The Molecular Taxonomy of Breast Cancer International Consortium (METABRIC).

    Gene and protein expression profiles to determine new breast cancer subgroups in
    order to help physicians provide better treatment recommendations.

    This is the processed data set used in the DeepSurv paper (Katzman et al. 2018), and details
    can be found at https://doi.org/10.1186/s12874-018-0482-1

    See https://github.com/jaredleekatzman/DeepSurv/tree/master/experiments/data
    for original data.
    """
    name = 'metabric'


class _Gbsg(_DatasetDeepSurv):
    """ Rotterdam & German Breast Cancer Study Group (GBSG)

    A combination of the Rotterdam tumor bank and the German Breast Cancer Study Group.

    This is the processed data set used in the DeepSurv paper (Katzman et al. 2018), and details
    can be found at https://doi.org/10.1186/s12874-018-0482-1

    See https://github.com/jaredleekatzman/DeepSurv/tree/master/experiments/data
    for original data.
    """
    name = 'gbsg'
