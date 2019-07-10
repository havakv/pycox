import warnings
import numpy as np
from sklearn.preprocessing import StandardScaler
from pycox.preprocessing.discretization import make_cuts, IdxDiscUnknownC, _values_if_series


class LabTransCoxTime:
    """
    Label transforms useful for CoxTime models, as we create 'map_scaled_to_orig' which
    is the inverse transform of the training data.

    Use it to e.g. set index of survival predictions:
        surv = cox_time.predict_survival_function(x_test)
        surv.index = labtrans.map_scaled_to_orig(surv.index)
    
    Keyword Arguments:
        log_duration {bool} -- Log transform duration, i.e. 'log(1+x)'. (default: {False})
        with_mean {bool} -- Center the duration before scaling.
            Passed to sklearn.preprocessing.StandardScaler (default: {True})
        with_std {bool} -- Scale duration to unit variance.
            Passed to sklearn.preprocessing.StandardScaler (default: {True})
    """
    def __init__(self, log_duration=False, with_mean=True, with_std=True):
        self.log_duration = log_duration
        self.duration_scaler = StandardScaler(True, with_mean, with_std)

    @property
    def map_scaled_to_orig(self):
        """Map from transformed durations back to the original durations, i.e. inverce transform.

        Use it to e.g. set index of survival predictions:
            surv = cox_time.predict_survival_function(x_test)
            surv.index = labtrans.map_scaled_to_orig(surv.index)
        """
        if not hasattr(self, '_inverse_duration_map'):
            raise ValueError('Need to fit the models before you can call this method')
        return self._inverse_duration_map

    def fit(self, durations, events):
        self.fit_transform(durations, events)
        return self

    def fit_transform(self, durations, events):
        train_durations = durations
        durations = durations.astype('float32')
        events = events.astype('float32')
        if self.log_duration:
            durations = np.log1p(durations)
        durations = self.duration_scaler.fit_transform(durations.reshape(-1, 1)).flatten()
        self._inverse_duration_map = {scaled: orig for orig, scaled in zip(train_durations, durations)}
        self._inverse_duration_map = np.vectorize(self._inverse_duration_map.get)
        return durations, events

    def transform(self, durations, events):
        durations = durations.astype('float32')
        events = events.astype('float32')
        if self.log_duration:
            durations = np.log1p(durations)
        durations = self.duration_scaler.transform(durations.reshape(-1, 1)).flatten()
        return durations, events


class LabTransDiscreteSurv:
    """
    Discretize continuous (duration, event) pairs based on cuts.
    One can either determine the cuts points in form of passing an array to this class,
    or one can obtain cut points based on the trainig data.

    The discretization learned from fitting to data will move censorings to the left cut point,
    and events to right cut point.

    Arguments:
        cuts {int, tuple, array} -- Defining cut points.
            If 'int' we create an equidistant grid with 'cuts' cut points,
            if 'array' we used these defined cut points,
            if 'tuple' with ('str', int) we call 'pycox.preprocessing.discretization.make_cuts'
            on trainig data set. (deafult: {20})
    
    Keyword Arguments:
        min_ {float} -- Starting duration (default: {0.})
        dtype {str, dtype} -- dtype of discretization.
    """
    def __init__(self, cuts, min_=0., dtype=None):
        self._predefined_cuts = False
        if type(cuts) is int:
            cuts = ('equidistant', cuts)
        elif hasattr(cuts, '__iter__'):
            if (type(cuts[0]) is not str):
                self.idu = IdxDiscUnknownC(cuts)
                assert dtype is None, "Need dtype to be None for spesified cuts"
                self.dtype = type(cuts[0])
                self._dtype = self.dtype
                self._predefined_cuts = True
        self._cuts = cuts
        self.min_ = min_
        self._dtype = dtype

    def fit(self, durations, events):
        if self._predefined_cuts:
            warnings.warn("Calling fit method, when 'cuts' are allready definded. Leaving cuts unchanges.")
            return self
        self.dtype = self._dtype
        if self.dtype is None:
            if isinstance(durations[0], np.floating):
                self.dtype = durations.dtype
            else:
                self.dtype = np.dtype('float64')
        durations = durations.astype(self.dtype)
        self.cuts = make_cuts(self._cuts, durations, events, self.min_, self.dtype)
        self.idu = IdxDiscUnknownC(self.cuts)
        return self

    def fit_transform(self, durations, events):
        self.fit(durations, events)
        idx_durations, events = self.transform(durations, events)
        return idx_durations, events

    def transform(self, durations, events):
        durations = _values_if_series(durations)
        durations = durations.astype(self.dtype)
        events = _values_if_series(events)
        idx_durations, events = self.idu.transform(durations, events)
        return idx_durations, events.astype('float32')

