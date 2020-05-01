from pathlib import Path
import pandas as pd
import pycox
import os

_DATA_OVERRIDE = os.environ.get('PYCOX_DATA_DIR', None)
if _DATA_OVERRIDE:
    _PATH_DATA = Path(_DATA_OVERRIDE)
else:
    _PATH_ROOT = Path(pycox.__file__).parent
    _PATH_DATA = _PATH_ROOT / 'datasets' / 'data'
_PATH_DATA.mkdir(parents=True, exist_ok=True)

class _DatasetLoader:
    """Abstract class for loading data sets.
    """
    name = NotImplemented
    _checksum = None

    def __init__(self):
        self.path = _PATH_DATA / f"{self.name}.feather"

    def read_df(self):
        if not self.path.exists():
            print(f"Dataset '{self.name}' not locally available. Downloading...")
            self._download()
            print(f"Done")
        df = pd.read_feather(self.path)
        df = self._label_cols_at_end(df)
        return df
    
    def _download(self):
        raise NotImplementedError
    
    def delete_local_copy(self):
        if not self.path.exists():
            raise RuntimeError("File does not exists.")
        self.path.unlink()

    def _label_cols_at_end(self, df):
        if hasattr(self, 'col_duration') and hasattr(self, 'col_event'):
            col_label = [self.col_duration, self.col_event]
            df = df[list(df.columns.drop(col_label)) + col_label]
        return df

    def checksum(self):
        """Checks that the dataset is correct. 
        
        Returns:
            bool -- If the check passed.
        """
        if self._checksum is None:
            raise NotImplementedError("No available comparison for this dataset.")
        df = self.read_df()
        return self._checksum_df(df)

    def _checksum_df(self, df):
        if self._checksum is None:
            raise NotImplementedError("No available comparison for this dataset.")
        import hashlib
        val = get_checksum(df)
        return val == self._checksum


def get_checksum(df):
    import hashlib
    val = hashlib.sha256(df.to_csv().encode()).hexdigest()
    return val
