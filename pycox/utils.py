'''
Some usefule methods to simplify use of methods.
'''

import numpy as np
import pandas as pd
from sklearn.pipeline import FeatureUnion
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper

class DataFrameFeatureUnion(TransformerMixin):
    '''Like sklearn.pipeline.FeatureUnion, but for DataFrameMapper in pandas_sklearn.
    '''
    def __init__(self, transformer_list, n_jobs=1, df_out=True):
        self.transformer_list = transformer_list
        self.n_jobs=1
        self.df_out=df_out
        self.mapper = FeatureUnion(self.transformer_list, n_jobs=self.n_jobs)
    
    def fit(self, X, y=None):
        self.mapper.fit(X, y)
        return self
    
    def transform(self, X, y=None):
        X_trans = self.mapper.transform(X)
        if self.df_out is False:
            return X_trans
        self.transformed_names_ = [name for trans in self.transformer_list 
                                   for name in trans[1].transformed_names_]
        return pd.DataFrame(X_trans, columns=self.transformed_names_, index=X.index)


class MapperCoxTime(object):
    '''Class for simplify working with our non-poportional cox model.
    As time is used as a covariate, it is nice to have methods for transforming back and forwards.
    This class transforms the time with `transform`, and applies StandardScaler.
    
    Parameters:
        cov_mapper: DataFrameMapper for covariates (excluding time/duration).
        duration_col: Name of duration column.
        event_col: Name of event column. If None, we don't use evnet_col.
        log: If we should log transform (np.log1p) durations before applying StandardScalar().
    
    Example:

    from sklearn_pandas import DataFrameMapper
    from sklearn.preprocessing import StandardScaler
    cov_mapper = DataFrameMapper([
        (['col_1'], StandardScaler()),
        (['col_2'], StandardScaler()),
    ])
    mapper = MapperCoxTime(cov_mapper, 'time', 'churn')
    df_train = mapper.fit_transform(train)
    df_test = mapper.transform(test)

    # To do inverse transform of time do
    mapper.inverse_duration(df_train['time'])
    '''
    def __init__(self, cov_mapper, duration_col, event_col=None, log=False):
        self.cov_mapper = cov_mapper
        self.duration_col = duration_col
        self.event_col = event_col
        self.log = log
        self._duration_scaler = StandardScaler()
        self._duration_mapper = DataFrameMapper([([self.duration_col], self._duration_scaler)])
        self.mapper = DataFrameFeatureUnion([('covariates', self.cov_mapper), ('duration', self._duration_mapper)],
                                            df_out=True)
   
    def _log_trans_duration(self, df):
        return df.assign(**{self.duration_col: np.log1p(df[self.duration_col])})
    
    def _add_event(self, df, df_with_event):
        return df.assign(**{self.event_col: df_with_event[self.event_col].values})

    def fit(self, df):
        if self.log:
            df = self._log_trans_duration(df)
        self.mapper.fit(df)
        return self
    
    def transform(self, df, drop_event=False):
        '''If drop_event, don't include the event col in the output.
        '''
        if self.log:
            df = self._log_trans_duration(df)
        df_trans = self.mapper.transform(df)
        if self.event_col and (drop_event is False):
            df_trans = self._add_event(df_trans, df)
        return df_trans
    
    def fit_transform(self, df, drop_event=False):
        '''If drop_event, don't include the event col in the output.
        '''
        return self.fit(df).transform(df, drop_event)

    def transform_duration(self, X):
        '''Transform only duration column.
        
        Parameters:
            X: pd.Seres, pd.DataFrame or np.ndarray with duration.
                If DataFrame, the duration column need to have correct name.
        '''
        if not hasattr(X, '__iter__'):
            X = np.array(X)
        if X.__class__ in [np.ndarray, pd.Series]:
            X = self._duration_single_to_df(X)
        if self.log:
            X = self._log_trans_duration(X)
        trans = self._duration_mapper.transform(X)
        return pd.Series(trans.flatten(), index=X.index)
    
    def _duration_single_to_df(self, x):
        if x.__class__ is np.ndarray:
            x = x.flatten()
        return pd.DataFrame(pd.Series(x).rename(self.duration_col))
        
    
    def inverse_duration(self, X):
        '''Invers transformation of duration column.
        
        Parameters:
            X: pd.Seres, pd.DataFrame or np.ndarray with duration.
                If DataFrame, the duration column need to have correct name.
        '''
        if X.__class__ is pd.DataFrame:
            X = X[self.duration_col]
        inverse = self._duration_scaler.inverse_transform(X)
        index = X.index if hasattr(X, 'index') else None
        inverse = pd.Series(inverse, index=index)
        if self.log:
            inverse = np.exp(inverse) - 1
        return inverse

# def to_cuda(obj, cuda_args):
#     '''Some general rules for using obj.cuda()

#     If cuda_args is True: obj.cuda().
#     If cuda_args is iterage obj.cuda(**cuda_args)
#     Else (typically int): obj.cuda(cuda_args).
#     '''
#     if cuda_args is True:
#         return obj.cuda()
#     if hasattr(cuda_args, '__iter__'):
#         return obj.cuda(**cuda_args)
#     return obj.cuda(cuda_args)