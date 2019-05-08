import numpy as np
from sklearn.preprocessing import StandardScaler


class LabTransCoxTime:
    def __init__(self, log_duration=False, with_mean=True, with_std=True):
        self.log_duration = log_duration
        self.duration_scaler = StandardScaler(True, with_mean, with_std)

    @property
    def map_scaled_to_orig(self):
        if not hasattr(self, '_inverse_duration_map'):
            raise ValueError('Need to fit the models before you can call this method')
        return self._inverse_duration_map

    def fit(self, durations, events):
        train_durations = durations
        duration = durations.astype('float32')
        if self.log_duration:
            durations = np.log1p(durations)
        durations = self.duration_scaler.fit_transform(durations.reshape(-1, 1)).flatten()
        self._inverse_duration_map = {scaled: orig for orig, scaled in zip(train_durations, durations)}
        self._inverse_duration_map = np.vectorize(self._inverse_duration_map.get)
        return self

    def fit_transform(self, durations, events):
        train_durations = durations
        duration = durations.astype('float32')
        events = events.astype('float32')
        if self.log_duration:
            durations = np.log1p(durations)
        durations = self.duration_scaler.fit_transform(durations.reshape(-1, 1)).flatten()
        self._inverse_duration_map = {scaled: orig for orig, scaled in zip(train_durations, durations)}
        self._inverse_duration_map = np.vectorize(self._inverse_duration_map.get)
        return durations, events

    def transform(self, durations, events):
        duration = durations.astype('float32')
        events = events.astype('float32')
        if self.log_duration:
            durations = np.log1p(durations)
        durations = self.duration_scaler.transform(durations.reshape(-1, 1)).flatten()
        return durations, events