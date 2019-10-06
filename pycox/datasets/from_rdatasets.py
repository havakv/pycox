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
        futime: (duration)
            days from enrollment until death. Note that there are 3 subjects whose sample
            was obtained on their death date.
        death: (event)
            0=alive at last contact date, 1=dead.
        chapter:
            for those who died, a grouping of their primary cause of death by chapter headings
            of the International Code of Diseases ICD-9.
    
    """
    name = 'flchain'
    col_duration = 'futime'
    col_event = 'death'
    _checksum = 'ec12748a1aa5790457c09793387337bb03b1dc45a22a2d58a8c2b9ad1f2648dd'

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


class _Nwtco(_DatasetRdatasetsSurvival):
    """Data from the National Wilm's Tumor Study (NWTCO)
    Obtained from Rdatasets (https://github.com/vincentarelbundock/Rdatasets).

    Measurement error example. Tumor histology predicts survival, but prediction is stronger
    with central lab histology than with the local institution determination.

    For details see http://vincentarelbundock.github.io/Rdatasets/doc/survival/nwtco.html

    Variables:
        seqno:
            id number
        instit:
            histology from local institution
        histol:
            histology from central lab
        stage:
            disease stage
        study:
            study
        rel: (event)
            indicator for relapse
        edrel: (duration)
            time to relapse
        age:
            age in months
        in.subcohort:
            included in the subcohort for the example in the paper

    References
        NE Breslow and N Chatterjee (1999), Design and analysis of two-phase studies with binary
        outcome applied to Wilms tumor prognosis. Applied Statistics 48, 457–68.
    """
    name = 'nwtco'
    col_duration = 'edrel'
    col_event = 'rel'
    _checksum = '5aa3de698dadb60154dd59196796e382739ff56dc6cbd39cfc2fda50d69d118e'

    def read_df(self, processed=True):
        """Get dataset.

        If 'processed' is False, return the raw data set.
        See the code for processing.
        
        Keyword Arguments:
            processed {bool} -- If 'False' get raw data, else get processed (see '??nwtco.read_df').
                (default: {True})
        """
        df = super().read_df()
        if processed:
            df = (df
                  .assign(instit_2=df['instit'] - 1,
                          histol_2=df['histol'] - 1,
                          study_4=df['study'] - 3,
                          stage=df['stage'].astype('category'))
                  .drop(['Unnamed: 0', 'seqno', 'instit', 'histol', 'study'], axis=1))
            for col in df.columns.drop('stage'):
                df[col] = df[col].astype('float32')
            df = self._label_cols_at_end(df)
        return df