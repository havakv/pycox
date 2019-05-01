import pandas as pd
from pycox.datasets._dataset_loader import _DatasetLoader

def download_from_rdatasets(package, name):
    datasets = (pd.read_csv("http://vincentarelbundock.github.com/Rdatasets/datasets.csv")
                .loc[lambda x: x['Package'] == package].set_index('Item'))
    if not name in datasets.index:
        raise ValueError(f"Dataset {name} not found.")
    info = datasets.loc[name]
    url = info.CSV 
    return pd.read_csv(url), info


class _DatasetRdatasetsSurvival(_DatasetLoader):
    """Data sets from Rdataset survival.
    """
    def _download(self):
        df, info = download_from_rdatasets('survival', self.name)
        self.info = info
        df.to_feather(self.path)


class _Flchain(_DatasetRdatasetsSurvival):
    """Assay of serum free light chain (FLCHAIN).
    Obtained from Rdatasets (https://github.com/vincentarelbundock/Rdatasets).

    A study of the relationship between serum free light chain (FLC) and mortality.
    The original sample contains samples on approximately 2/3 of the residents of Olmsted
    County aged 50 or greater.

    For details see http://vincentarelbundock.github.io/Rdatasets/doc/survival/flchain.html

    Variables:
        age:
            age in years.
        sex:
            F=female, M=male.
        sample.yr:
            the calendar year in which a blood sample was obtained.
        kappa:
            serum free light chain, kappa portion.
        lambda:
            serum free light chain, lambda portion.
        flc.grp:
            the FLC group for the subject, as used in the original analysis.
        creatinine:
            serum creatinine.
        mgus:
            1 if the subject had been diagnosed with monoclonal gammapothy (MGUS).
        futime:
            days from enrollment until death. Note that there are 3 subjects whose sample
            was obtained on their death date.
        death:
            0=alive at last contact date, 1=dead.
        chapter:
            for those who died, a grouping of their primary cause of death by chapter headings
            of the International Code of Diseases ICD-9.
    
    """
    name = 'flchain'
    def read_df(self, processed=True):
        """Get dataset.

        If 'processed' is False, return the raw data set.
        See the code for processing.
        
        Keyword Arguments:
            processed {bool} -- If 'False' get raw data, else get processed (see '??flchain.read_df').
                (default: {True})
        """
        df = super().read_df()
        if processed:
            df = (df
                  .drop(['chapter', 'Unnamed: 0'], axis=1)
                  .loc[lambda x: x['creatinine'].isna() == False]
                  .reset_index(drop=True)
                  .assign(sex=lambda x: (x['sex'] == 'M')))

            categorical = ['sample.yr', 'flc.grp']
            for col in categorical:
                df[col] = df[col].astype('category')
            for col in df.columns.drop(categorical):
                df[col] = df[col].astype('float32')
        return df

