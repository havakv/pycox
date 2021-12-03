import warnings
import numpy as np
from sklearn.preprocessing import StandardScaler
from pycox.preprocessing.discretization import (make_cuts, IdxDiscUnknownC, _values_if_series,
    DiscretizeUnknownC, Duration2Idx)


class LabTransCoxTime:
    """
    Label transforms useful for CoxTime models. It can log-transform and standardize the durations.

    It also creates `map_scaled_to_orig` which is the inverse transform of the durations data,
    enabling us to set the correct time scale for predictions.
    This can be done by passing the object to the CoxTime init:
        model = CoxTime(net, labrans=labtrans)
    which gives the correct time scale of survival predictions
        surv = model.predict_surv_df(x)
    
    Keyword Arguments:
        log_duration {bool} -- Log-transform durations, i.e. 'log(1+x)'. (default: {False})
        with_mean {bool} -- Center the duration before scaling.
            Passed to `sklearn.preprocessing.StandardScaler` (default: {True})
        with_std {bool} -- Scale duration to unit variance.
            Passed to `sklearn.preprocessing.StandardScaler` (default: {True})
    """
    def __init__(self, log_duration=False, with_mean=True, with_std=True):
        self.log_duration = log_duration
        self.duration_scaler = StandardScaler(copy=True, with_mean=with_mean, with_std=with_std)

    @property
    def map_scaled_to_orig(self):
        """Map from transformed durations back to the original durations, i.e. inverse transform.

        Use it to e.g. set index of survival predictions:
            surv = model.predict_surv_df(x_test)
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

    @property
    def out_features(self):
        """Returns the number of output features that should be used in the torch model.
        This always returns 1, and is just included for api design purposes.
        
        Returns:
            [int] -- Number of output features.
        """
        return 1


class LabTransDiscreteTime:
    """
    Discretize continuous (duration, event) pairs based on a set of cut points.
    One can either determine the cut points in form of passing an array to this class,
    or one can obtain cut points based on the training data.

    The discretization learned from fitting to data will move censorings to the left cut point,
    and events to right cut point.

    Arguments:
        cuts {int, array} -- Defining cut points, either the number of cuts, or the actual cut points.
    
    Keyword Arguments:
        scheme {str} -- Scheme used for discretization. Either 'equidistant' or 'quantiles'
            (default: {'equidistant})
        min_ {float} -- Starting duration (default: {0.})
        dtype {str, dtype} -- dtype of discretization.
    """
    def __init__(self, cuts, scheme='equidistant', min_=0., dtype=None):
        self._cuts = cuts
        self._scheme = scheme
        self._min = min_
        self._dtype_init = dtype
        self._predefined_cuts = False
        self.cuts = None
        if hasattr(cuts, '__iter__'):
            if type(cuts) is list:
                cuts = np.array(cuts)
            self.cuts = cuts
            self.idu = IdxDiscUnknownC(self.cuts)
            assert dtype is None, "Need `dtype` to be `None` for specified cuts"
            self._dtype = type(self.cuts[0])
            self._dtype_init = self._dtype
            self._predefined_cuts = True

    def fit(self, durations, events):
        if self._predefined_cuts:
            warnings.warn("Calling fit method, when 'cuts' are already defined. Leaving cuts unchanged.")
            return self
        self._dtype = self._dtype_init
        if self._dtype is None:
            if isinstance(durations[0], np.floating):
                self._dtype = durations.dtype
            else:
                self._dtype = np.dtype('float64')
        durations = durations.astype(self._dtype)
        self.cuts = make_cuts(self._cuts, self._scheme, durations, events, self._min, self._dtype)
        self.idu = IdxDiscUnknownC(self.cuts)
        return self

    def fit_transform(self, durations, events):
        self.fit(durations, events)
        idx_durations, events = self.transform(durations, events)
        return idx_durations, events

    def transform(self, durations, events):
        durations = _values_if_series(durations)
        durations = durations.astype(self._dtype)
        events = _values_if_series(events)
        idx_durations, events = self.idu.transform(durations, events)
        return idx_durations.astype('int64'), events.astype('float32')

    @property
    def out_features(self):
        """Returns the number of output features that should be used in the torch model.
        
        Returns:
            [int] -- Number of output features.
        """
        if self.cuts is None:
            raise ValueError("Need to call `fit` before this is accessible.")
        return len(self.cuts)


class LabTransPCHazard:
    """
    Defining time intervals (`cuts`) needed for the `PCHazard` method [1].
    One can either determine the cut points in form of passing an array to this class,
    or one can obtain cut points based on the training data.

    Arguments:
        cuts {int, array} -- Defining cut points, either the number of cuts, or the actual cut points.
    
    Keyword Arguments:
        scheme {str} -- Scheme used for discretization. Either 'equidistant' or 'quantiles'
            (default: {'equidistant})
        min_ {float} -- Starting duration (default: {0.})
        dtype {str, dtype} -- dtype of discretization.

    References:
    [1] Håvard Kvamme and Ørnulf Borgan. Continuous and Discrete-Time Survival Prediction
        with Neural Networks. arXiv preprint arXiv:1910.06724, 2019.
        https://arxiv.org/pdf/1910.06724.pdf
    """
    def __init__(self, cuts, scheme='equidistant', min_=0., dtype=None):
        self._cuts = cuts
        self._scheme = scheme
        self._min = min_
        self._dtype_init = dtype
        self._predefined_cuts = False
        self.cuts = None
        if hasattr(cuts, '__iter__'):
            if type(cuts) is list:
                cuts = np.array(cuts)
            self.cuts = cuts
            self.idu = IdxDiscUnknownC(self.cuts)
            assert dtype is None, "Need `dtype` to be `None` for specified cuts"
            self._dtype = type(self.cuts[0])
            self._dtype_init = self._dtype
            self._predefined_cuts = True
        else:
            self._cuts += 1

    def fit(self, durations, events):
        if self._predefined_cuts:
            warnings.warn("Calling fit method, when 'cuts' are already defined. Leaving cuts unchanged.")
            return self
        self._dtype = self._dtype_init
        if self._dtype is None:
            if isinstance(durations[0], np.floating):
                self._dtype = durations.dtype
            else:
                self._dtype = np.dtype('float64')
        durations = durations.astype(self._dtype)
        self.cuts = make_cuts(self._cuts, self._scheme, durations, events, self._min, self._dtype)
        self.duc = DiscretizeUnknownC(self.cuts, right_censor=True, censor_side='right')
        self.di = Duration2Idx(self.cuts)
        return self

    def fit_transform(self, durations, events):
        self.fit(durations, events)
        return self.transform(durations, events)

    def transform(self, durations, events):
        durations = _values_if_series(durations)
        durations = durations.astype(self._dtype)
        events = _values_if_series(events)
        dur_disc, events = self.duc.transform(durations, events)
        idx_durations = self.di.transform(dur_disc)
        cut_diff = np.diff(self.cuts)
        assert (cut_diff > 0).all(), 'Cuts are not unique.'
        t_frac = 1. - (dur_disc - durations) / cut_diff[idx_durations-1]
        if idx_durations.min() == 0:
            warnings.warn("""Got event/censoring at start time. Should be removed! It is set s.t. it has no contribution to loss.""")
            t_frac[idx_durations == 0] = 0
            events[idx_durations == 0] = 0
        idx_durations = idx_durations - 1
        return idx_durations.astype('int64'), events.astype('float32'), t_frac.astype('float32')

    @property
    def out_features(self):
        """Returns the number of output features that should be used in the torch model.
        
        Returns:
            [int] -- Number of output features.
        """
        if self.cuts is None:
            raise ValueError("Need to call `fit` before this is accessible.")
        return len(self.cuts) - 1
