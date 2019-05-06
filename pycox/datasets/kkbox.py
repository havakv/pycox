import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from pycox.datasets._dataset_loader import _DatasetLoader, _PATH_DATA

class _DatasetKKBoxChurn(_DatasetLoader):
    """KKBox churn data set

    To make it identical to the data set in Kvamme et al. (2019) we need apply the log transfrom 
    'z = log(x - min(x) + 1)' to the variables
    ['actual_amount_paid', 'days_between_subs', 'days_since_reg_init', 'payment_plan_days', 'plan_list_price'].
    """
    name = 'kkbox'
    def __init__(self):
        self._path_dir = _PATH_DATA / self.name
        self.path_train = self._path_dir / 'train.feather'
        self.path_test = self._path_dir / 'test.feather'
        self.path_val = self._path_dir / 'val.feather'
        self.log_cols = ['actual_amount_paid', 'days_between_subs', 'days_since_reg_init',
                         'payment_plan_days', 'plan_list_price']


    def read_df(self, subset='train', log_trans=True):
        """Get train, test, val or survival dataset.

        The columns: 'duration' and 'event' gives the duration time and event indicator.

        The survival dataset contrains no covariates, but can be useful for extending
        the dataset with more covariates from Kaggle.

        If 'log_trans' is True, the columns in 'kkbox.log_cols' are log-transformed with 
        'z = log(x - min(x) + 1)'.
        """
        if subset == 'train':
            path = self.path_train
        elif subset == 'test':
            path = self.path_test
        elif subset == 'val':
            path = self.path_val
        elif subset == 'survival':
            path = self._path_dir / 'survival_data.feather'
        else:
            raise ValueError(f"Need 'subset' to be 'train', 'val', or 'test'. Got {subset}")
        
        if not path.exists():
            print(f"""
            The KKBox dataset not locally available.
            If you want to download, call 'kkbox.download_kkbox()', but note that 
            this might take a LONG TIME!!!
            NOTE: You need kaggle credentials! Follow instructions at 
            https://github.com/Kaggle/kaggle-api#api-credentials
            """)
            return None

        def log_min_p(col, df):
            x = df[col]
            min_ = -1. if col == 'days_since_reg_init' else 0.
            return np.log(x - min_ + 1)

        df = pd.read_feather(path)
        if log_trans:
            df = df.assign(**{col: log_min_p(col, df) for col in self.log_cols})
            df = df.rename(columns={col: f"log_{col}" for col in self.log_cols})
        return df


    def download_kkbox(self):
        """Download KKBox data set. 
        This is likey to take a LONG time!!!
        NOTE: You need kaggle credentials! Follow instructions at 
        https://github.com/Kaggle/kaggle-api#api-credentials
        """
        self._download()

    def _download(self):
        self._setup_download_dir()
        self._7z_from_kaggle()
        self._csv_to_feather_with_types()
        print('Creating survival data...')
        self._make_survival_data()
        print('Creating covariates...')
        self._make_survival_covariates()
        print('Creating train/test/val subsets...')
        self._make_train_test_split()
        print('Cleaning up...')
        self._clean_up()
        print('Done!')

    def _setup_download_dir(self):
        if self._path_dir.exists():
            self._clean_up()
            # if self._path_dir.is_dir():
            #     for file in self._path_dir.iterdir():
            #         try:
            #             file.unlink()
            #         except IsADirectoryError:
            #             warnings.warn(f"Encountered directory in {self._path_dir}")
            # else:
            #     raise OSError(f"'{self._path_dir}' allready exists and is not a directory. Something wrong.'")
        else:
            self._path_dir.mkdir()

    def _7z_from_kaggle(self):
        import subprocess
        try:
            import kaggle
        except OSError as e:
            raise OSError(
            f""""
            Need to provide kaggle credentials to download this data set. See guide at
            https://github.com/Kaggle/kaggle-api#api-credentials.
            """
            )
        # files =  ['train.csv.7z', 'transactions.csv.7z', 'members_v3.csv.7z']
        # files =  ['train.csv.7z']
        files =  ['train', 'transactions', 'members_v3']
        print('Downloading from kaggle...')
        for file in files:
            kaggle.api.competition_download_file('kkbox-churn-prediction-challenge', file + '.csv.7z',
                                                 path=self._path_dir, force=True)
        for file in files:
            print(f"Extracting '{file}'...")
            subprocess.check_output(['7z',  'x', str(self._path_dir / (file + '.csv.7z')),
                                     f"-o{self._path_dir}"])
            print(f"Finished extracting '{file}'.")
        # subprocess.check_output(['mv',  path_dir / 'members_v3.csv', path_dir / 'members.csv'])

    def _csv_to_feather_with_types(self):
        print("Making feather data frames...")
        file = 'train'
        pd.read_csv(self._path_dir / f"{file}.csv").to_feather(self._path_dir / f"{file}_raw.feather")

        file = 'members'
        members = pd.read_csv(self._path_dir / f"{file}_v3.csv",
                              parse_dates=['registration_init_time'])
        (members.assign(**{col: members[col].astype('category')
                           for col in ['city', 'registered_via', 'gender']})
         .to_feather(self._path_dir / f"{file}.feather"))

        file = 'transactions'
        trans = pd.read_csv(self._path_dir / f"{file}.csv", parse_dates=['transaction_date', 'membership_expire_date'])
        (trans.assign(**{col: trans[col].astype('category') for col in ['payment_method_id', 'is_auto_renew', 'is_cancel']})
         .to_feather(self._path_dir / f"{file}.feather"))

    def _make_survival_data(self):
        """Comine the downloaded files and create a survival data sets
        (more or less without covariates).

        A customer is considered churned if one of the following is true:
            - If it has been more than 30 days since the expiration data of a membership subscription until the next transaction.
            - If the customer has expiration in before 2017-03-01, and no transaction after that.
        """
        train = pd.read_feather(self._path_dir / 'train_raw.feather')
        members = pd.read_feather(self._path_dir / 'members.feather')
        trans = (pd.read_feather(self._path_dir / 'transactions.feather')
                 [['msno', 'transaction_date', 'membership_expire_date', 'is_cancel']])
        last_churn_date = '2017-01-29' # 30 days before last transactions are made in the dataset.

        # Chunr: More than 30 days before reentering
        def days_without_membership(df):
            diff = (df['next_trans_date'] - df['membership_expire_date']).dt.total_seconds()
            return diff / (60 * 60 * 24)
        
        trans = (trans
                 .sort_values(['msno', 'transaction_date'])
                 .assign(next_trans_date=(lambda x: x.groupby('msno')['transaction_date'].shift(-1)))
                 .assign(churn30=lambda x: days_without_membership(x) > 30))
        
        # Remove entries with membership_expire_date < transaction_date
        trans = trans.loc[lambda x: x['transaction_date'] <= x['membership_expire_date']]
        assert (trans.loc[lambda x: x['churn30']==True].groupby(['msno', 'transaction_date'])['msno'].count().max() == 1)

        # Churn: Leaves forever
        trans = (trans
                 .assign(max_trans_date=lambda x: x.groupby('msno')['transaction_date'].transform('max'))
                 .assign(final_churn=(lambda x:
                                     (x['max_trans_date'] <= last_churn_date) &
                                     (x['transaction_date'] == x['max_trans_date']) & 
                                     (x['membership_expire_date'] <= last_churn_date)
                                     )))
        
        # Churn: From training set
        trans = (trans
                 .merge(train, how='left', on='msno')
                 .assign(train_churn=lambda x: x['is_churn'].fillna(0).astype('bool'))
                 .drop('is_churn', axis=1)
                 .assign(train_churn=lambda x: (x['max_trans_date'] == x['transaction_date']) & x['train_churn'])
                 .assign(churn=lambda x: x['train_churn'] | x['churn30'] | x['final_churn']))
        
        # Split individuals on churn
        trans = (trans
                 .join(trans
                       .sort_values(['msno', 'transaction_date'])
                       .groupby('msno')[['churn30', 'membership_expire_date']].shift(1)
                       .rename(columns={'churn30': 'new_start', 'membership_expire_date': 'prev_mem_exp_date'})))
        
        def number_of_new_starts(df):
            return (df
                    .assign(new_start=lambda x: x['new_start'].astype('float'))
                    .sort_values(['msno', 'transaction_date'])
                    .groupby('msno')
                    ['new_start'].cumsum().fillna(0.)
                    .astype('int'))
        
        def days_between_subs(df):
            diff = (df['transaction_date'] - df['prev_mem_exp_date']).dt
            diff = diff.total_seconds() / (60 * 60 * 24)
            df = df.assign(days_between_subs=diff)
            df.loc[lambda x: x['new_start'] != True, 'days_between_subs'] = np.nan
            return df['days_between_subs']

        trans = (trans
                 .assign(n_prev_churns=lambda x: number_of_new_starts(x),
                         days_between_subs=lambda x: days_between_subs(x)))
        
        # Set start times
        trans = (trans
                 .assign(start_date=trans.groupby(['msno', 'n_prev_churns'])['transaction_date'].transform('min'))
                 .assign(first_churn=lambda x: (x['n_prev_churns'] == 0) & (x['churn'] == True)))
    
        # Get only last transactions (per chrun)
        indivs = (trans
                  .assign(censored=lambda x: x.groupby('msno')['churn'].transform('sum') == 0)
                  .assign(last_censored=(lambda x: 
                                          (x['censored'] == True) &
                                          (x['transaction_date'] == x['max_trans_date'])
                                          ))
                  .loc[lambda x: x['last_censored'] | x['churn']]
                  .merge(members[['msno', 'registration_init_time']], how='left', on='msno'))
        
        def time_diff_days(df, last, first):
            return (df[last] - df[first]).dt.total_seconds() / (60 * 60 * 24)

        indivs = (indivs
                  .assign(time=lambda x: time_diff_days(x, 'membership_expire_date', 'start_date'),
                          days_since_reg_init=lambda x: time_diff_days(x, 'start_date', 'registration_init_time')))

        # When multiple transactions on last day, remove all but the last
        indivs = indivs.loc[lambda x: x['transaction_date'] != x['next_trans_date']] 
        assert indivs.shape == indivs.drop_duplicates(['msno', 'transaction_date']).shape
        assert (indivs['churn'] != indivs['censored']).all()

        # Clean up and remove variables that are not from the first transaction day
        dropcols = ['transaction_date', 'is_cancel', 'next_trans_date', 'max_trans_date', 'prev_mem_exp_date',
                    'censored', 'last_censored', 'churn30', 'final_churn', 'train_churn', 'membership_expire_date']

        indivs = (indivs
                  .assign(churn_type=lambda x: 1*x['churn30'] + 2*x['final_churn'] + 4*x['train_churn'])
                  .assign(churn_type=lambda x: 
                          np.array(['censoring', '30days', 'final', '30days_and_final', 'train', 'train_and_30', 
                                      'train_and_final', 'train_30_and_final'])[x['churn_type']])
                  .drop(dropcols, axis=1))

        indivs = indivs.loc[lambda x: x['churn_type'] != 'train_30_and_final']
        indivs = indivs.loc[lambda x: x['time'] > 0]

        def as_category(df, columns):
            return df.assign(**{col: df[col].astype('category') for col in columns})

        def as_int(df, columns):
            return df.assign(**{col: df[col].astype('int') for col in columns})

        indivs = (indivs
                .pipe(as_int, ['time'])
                .pipe(as_category, ['new_start', 'churn_type']))

        indivs.reset_index(drop=True).to_feather(self._path_dir / 'survival_data.feather')

    def _make_survival_covariates(self):
        individs = pd.read_feather(self._path_dir / 'survival_data.feather')
        members = pd.read_feather(self._path_dir / 'members.feather')
        trans = (individs
                 .merge(pd.read_feather(self._path_dir / 'transactions.feather'), 
                         how='left', left_on=['msno', 'start_date'], right_on=['msno', 'transaction_date'])
                 .drop('transaction_date', axis=1) # same as start_date
                 .drop_duplicates(['msno', 'start_date'], keep='last') # keep last transaction on start_date (by idx)
                )
        assert trans.shape[0] == individs.shape[0]

        def get_age_at_start(df):
            fixed_date = pd.datetime(2017, 3, 1)
                # Not important what the date is, though it is reasonalbe to use the last.
            age_diff = (fixed_date - df['start_date']).dt.total_seconds() / (60*60*24*365)
            return np.round(df['bd'] - age_diff)
    
        trans = (trans
                 .merge(members.drop(['registration_init_time'], axis=1), how='left', on='msno')
                 .assign(age_at_start=lambda x: get_age_at_start(x))
                 .drop(['bd'], axis=1)
                 .assign(strange_age=lambda x: (x['age_at_start'] <= 0) | (x['age_at_start'] >= 100),
                         age_at_start=lambda x: x['age_at_start'].clip(lower=0, upper=100)))

        # days_beteen_subs 
        # There are None for (not new start), so we can just set them to zero, and we don't need to include another variable (as it allready exists).
        trans = trans.assign(days_between_subs=lambda x: x['days_between_subs'].fillna(0.))

        # days_since_reg_init 
        # We remove negative entries, set Nans to -1, and add a categorical value for missing.
        pd.testing.assert_frame_equal(trans.loc[lambda x: x['days_since_reg_init'].isnull()],
                                    trans.loc[lambda x: x['age_at_start'].isnull()])
        assert (members.registration_init_time.isnull() == members.bd.isnull()).all()

        trans = (trans
                 .loc[lambda x: (x['days_since_reg_init'] >= 0) | x['days_since_reg_init'].isnull()]
                 .assign(nan_days_since_reg_init=lambda x: x['days_since_reg_init'].isnull())
                 .assign(days_since_reg_init=lambda x: x['days_since_reg_init'].fillna(-1)))

        # age_at_start 
        # This is Nan when days_since_reg_init is nan. This is because registration_init_time is nan when bd is nan.
        # We have removed negative entries, so we set Nans to -1, but don't add dymmy because its eaqual to days_since_reg_init dummy.
        trans = trans.assign(age_at_start=lambda x: x['age_at_start'].fillna(-1.))

        # First churn variable
        # The first_churn variable is false for everyone that does not churn. Therefore we can't use it.
        # We use n_prev_churns == 0 instead
        trans = (trans
                 .drop('first_churn', axis=1)
                 .assign(no_prev_churns=lambda x: x['n_prev_churns'] == 0))

        # Drop variables that are not useful 
        trans = (trans
                 .drop(['start_date', 'registration_init_time', 'churn_type', 
                         'membership_expire_date', 'new_start'],
                     axis=1))

        # Remove payment_method_id
        # We could use this covariate, but we choose not to...
        trans = trans.drop('payment_method_id', axis=1)

        # ### Log transfrom variables
        # log_cols = ['actual_amount_paid', 'days_between_subs', 'days_since_reg_init', 'payment_plan_days',
        #             'plan_list_price']
        
        # log_min_p = lambda x: np.log(x - x.min() + 1)
        # trans_log = trans.assign(**{col: log_min_p(trans[col]) for col in self.log_cols})
        # assert trans_log[self.log_cols].pipe(np.isfinite).all().all()
        trans_log = trans

        trans_log = trans_log.rename(columns=dict(churn='event', time='duration'))
        float_cols =  ['n_prev_churns', 'days_between_subs', 'days_since_reg_init', 'payment_plan_days',
                    'plan_list_price', 'age_at_start', 'actual_amount_paid', 'is_auto_renew',
                    'is_cancel', 'strange_age', 'nan_days_since_reg_init', 'no_prev_churns', 'duration', 'event']
        trans_log = trans_log.assign(**{col: trans_log[col].astype('float32') for col in float_cols}) 
        # cov_file = join(data_dir, 'covariates.feather')
        trans_log.reset_index(drop=True).to_feather(self._path_dir / 'covariates.feather')

    def _make_train_test_split(self, seed=1234):
        from sklearn.model_selection import train_test_split
        np.random.seed(seed)
        covariates = pd.read_feather(self._path_dir / 'covariates.feather')

        def train_test_split_customer(df, col_customer, test_size):
            tr, te = train_test_split(df[[col_customer]].drop_duplicates(), test_size=test_size)
            train =  df.merge(tr, how='right', on=col_customer)
            test =  df.merge(te, how='right', on=col_customer)
            return train, test

        train, test = train_test_split_customer(covariates, 'msno', 0.25)
        train, val = train_test_split_customer(train, 'msno', 0.1)

        assert train.merge(test, how='inner', on='msno').shape[0] == 0
        assert train.merge(val, how='inner', on='msno').shape[0] == 0
        assert test.merge(val, how='inner', on='msno').shape[0] == 0

        train.to_feather(self._path_dir / 'train.feather')
        test.to_feather(self._path_dir / 'test.feather')
        val.to_feather(self._path_dir / 'val.feather')

    def _clean_up(self):
        remove = ['covariates.feather', 'train.csv.7z', 'transactions.csv.7z', 'members_v3.csv.7z',
                  'train.csv', 'transactions.csv', 'members_v3.csv',
                  'train_raw.feather', 'transactions.feather', 'members.feather']
        for file in self._path_dir.iterdir():
            if file.name not in remove:
                continue
            try:
                file.unlink()
            except IsADirectoryError:
                warnings.warn(f"Encountered directory in {self._path_dir}")
    
    def delete_local_copy(self):
        for path in [self.path_train, self.path_test, self.path_val]:
            path.unlink()
