import warnings
try:
    from .utils import MapperCoxTime
except:
    warnings.warn('Need sklearn_pandas to get MapperCoxTime')
