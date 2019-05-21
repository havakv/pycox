from pathlib import Path
import pandas as pd
import pycox

_PATH_ROOT = Path(pycox.__file__).parent.parent
_PATH_DATA = _PATH_ROOT / 'datasets'

class _DatasetLoader:
    """Abstract class for loading data sets.
    """
    name = NotImplemented
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
