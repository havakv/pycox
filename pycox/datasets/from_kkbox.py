import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from pycox.datasets._dataset_loader import _DatasetLoader, _PATH_DATA

class _DatasetKKBoxChurn(_DatasetLoader):
    """KKBox churn data set obtained from Kaggle (WSDM - KKBox's Churn Prediction Challenge 2017).
    https://www.kaggle.com/c/kkbox-churn-prediction-challenge/data
    This is the version of the data set presented by Kvamme et al. (2019) [1], but the preferred version
    is the `kkbox` version which included administrative censoring labels and and extra categorical variable.

    Requires installation of the Kaggle API (https://github.com/Kaggle/kaggle-api), 
    with credentials (https://github.com/Kaggle/kaggle-api).

    The data set contains churn information from KKBox, an Asian music streaming service. Churn is
    defined by a customer failing to obtain a new valid service subscription within 30 days after
    the current membership expires.
    This version of the data set only consider part of the information made available in the challenge,
    as it is intended to compare survival methods, and not compete in the challenge.

    The data set is split in train, test and validations, based on an individual's id ('msno').

    Variables:
        msno:
            Identifier for individual. An individual might churn multiple times.
        event:
            Churn indicator, 1: churn, 0: censoring.
        n_prev_churns:
            Number of previous churns by the individual.
        (log_)days_between_subs:
            Number of days between this and the last subscription (log-transformed), if previously
            churned.
        duration:
            Durations until churn or censoring.
        (log_)days_since_reg_init:
            Number of days since first registration (log-transformed).
        (log_)payment_plan_days:
            Number of days until current subscription expires (log-transformed).
        (log_)plan_list_price:
            Listed price of current subscription (log-transformed).
        (log_)actual_amount_paid:
            The amount payed for the subscription (log-transformed).
        is_auto_renew:
            Not explained in competition https://www.kaggle.com/c/kkbox-churn-prediction-challenge/data
        is_cancel:
            If the customer has canceled the subscription. Subscription cancellation does not imply the
            user has churned. A user may cancel service subscription due to change of service plans or
            other reasons. 
        city:
            City of customer.
        gender:
            Gender of customer.
        registered_via:
            Registration method.
        age_at_start:
            Age at beginning of subscription.
        strange_age:
            Indicator for strange ages.
        nan_days_since_reg_init:
            Indicator that we don't know when the customer first subscribed.
        no_prev_churns:
            Indicator if the individual has not previously churned.

    References:
        [1] Håvard Kvamme, Ørnulf Borgan, and Ida Scheel. Time-to-event prediction with neural networks
            and Cox regression. Journal of Machine Learning Research, 20(129):1–30, 2019.
            http://jmlr.org/papers/v20/18-424.html
    """
    name = 'kkbox_v1'
    _checksum = '705ca57c7efd2d916f0da2dd6e3a399b3b279773271d595c2f591fcf7bb7cae6'

    def __init__(self):
        self._path_dir = _PATH_DATA / self.name
        self.path_train = self._path_dir / 'train.feather'
        self.path_test = self._path_dir / 'test.feather'
        self.path_val = self._path_dir / 'val.feather'
        self.path_survival = self._path_dir / 'survival_data.feather'
        self.log_cols = ['actual_amount_paid', 'days_between_subs', 'days_since_reg_init',
                         'payment_plan_days', 'plan_list_price']


    def read_df(self, subset='train', log_trans=True):
        """Get train, test, val or general survival data set.

        The columns: 'duration' and 'event' gives the duration time and event indicator.

        The survival data set contains no covariates, but can be useful for extending
        the dataset with more covariates from Kaggle.
    
        Keyword Arguments:
            subset {str} -- Which subset to use ('train', 'val', 'test').
                Can also set 'survival' which will give df with survival information without
                covariates. (default: {'train'})
            log_trans {bool} -- If covariates in 'kkbox_v1.log_cols' (from Kvamme paper) should be
                transformed with 'z = log(x - min(x) + 1)'. (default: {True})
        """
        if subset == 'train':
            path = self.path_train
        elif subset == 'test':
            path = self.path_test
        elif subset == 'val':
            path = self.path_val
        elif subset == 'survival':
            path = self.path_survival
            return pd.read_feather(path)
        else:
            raise ValueError(f"Need 'subset' to be 'train', 'val', or 'test'. Got {subset}")
        
        if not path.exists():
            print(f"""
            The KKBox dataset not locally available.
            If you want to download, call 'kkbox_v1.download_kkbox()', but note that 
            this might take around 10 min!
            NOTE: You need Kaggle credentials! Follow instructions at 
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
        This is likely to take around 10 min!!!
        NOTE: You need Kaggle credentials! Follow instructions at 
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
        print("Done! You can now call `df = kkbox_v1.read_df()`.")

    def _setup_download_dir(self):
        if self._path_dir.exists():
            self._clean_up()
        else:
            self._path_dir.mkdir()

    def _7z_from_kaggle(self):
        import subprocess
        try:
            import kaggle
        except OSError as e:
            raise OSError(
            f""""
            Need to provide Kaggle credentials to download this data set. See guide at
            https://github.com/Kaggle/kaggle-api#api-credentials.
            """
            )
        files =  ['train', 'transactions', 'members_v3']
        print('Downloading from Kaggle...')
        for file in files:
            kaggle.api.competition_download_file('kkbox-churn-prediction-challenge', file + '.csv.7z',
                                                 path=self._path_dir, force=True)
        for file in files:
            print(f"Extracting '{file}'...")
            subprocess.check_output(['7z',  'x', str(self._path_dir / (file + '.csv.7z')),
                                     f"-o{self._path_dir}"])
            print(f"Finished extracting '{file}'.")

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
        """Combine the downloaded files and create a survival data sets
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

        # Churn: More than 30 days before reentering
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

        # indivs.reset_index(drop=True).to_feather(self._path_dir / 'survival_data.feather')
        indivs.reset_index(drop=True).to_feather(self.path_survival)

    def _make_survival_covariates(self):
        # individs = pd.read_feather(self._path_dir / 'survival_data.feather')
        individs = pd.read_feather(self.path_survival)
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

        # days_between_subs 
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
        # We have removed negative entries, so we set Nans to -1, but don't add dummy because its equal to days_since_reg_init dummy.
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

        # ### Log transform variables
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
        for path in [self.path_train, self.path_test, self.path_val, self.path_survival]:
            path.unlink()


class _DatasetKKBoxAdmin(_DatasetKKBoxChurn):
    """KKBox churn data set obtained from Kaggle (WSDM - KKBox's Churn Prediction Challenge 2017).
    https://www.kaggle.com/c/kkbox-churn-prediction-challenge/data
    This is the data set used in [1] and is an updated verion of `kkbox_v1` in [2], as it contains
    administrative censoring times and the categorical variable `payment_method_id`.
    All customers are administratively censored at the date 2017-01-29.

    This is the version of the data set presented by Kvamme and Borgan (2019) [1].

    Requires installation of the Kaggle API (https://github.com/Kaggle/kaggle-api), 
    with credentials (https://github.com/Kaggle/kaggle-api).

    The data set contains churn information from KKBox, an Asian music streaming service. Churn is
    defined by a customer failing to obtain a new valid service subscription within 30 days after
    the current membership expires.
    This version of the data set only consider part of the information made available in the challenge,
    as it is intended to compare survival methods, and not compete in the challenge.

    Variables:
        msno:
            Identifier for individual. An individual might churn multiple times.
        event:
            Churn indicator, 1: churn, 0: censoring.
        n_prev_churns:
            Number of previous churns by the individual.
        (log_)days_between_subs:
            Number of days between this and the last subscription (log-transformed), if previously
            churned.
        duration:
            Durations until churn or censoring.
        censor_duration:
            The administrative censoring times.
        (log_)days_since_reg_init:
            Number of days since first registration (log-transformed).
        (log_)payment_plan_days:
            Number of days until current subscription expires (log-transformed).
        (log_)plan_list_price:
            Listed price of current subscription (log-transformed).
        (log_)actual_amount_paid:
            The amount payed for the subscription (log-transformed).
        is_auto_renew:
            Not explained in competition https://www.kaggle.com/c/kkbox-churn-prediction-challenge/data
        is_cancel:
            If the customer has canceled the subscription. Subscription cancellation does not imply the
            user has churned. A user may cancel service subscription due to change of service plans or
            other reasons. 
        city:
            City of customer.
        gender:
            Gender of customer.
        registered_via:
            Registration method.
        age_at_start:
            Age at beginning of subscription.
        strange_age:
            Indicator for strange ages.
        nan_days_since_reg_init:
            Indicator that we don't know when the customer first subscribed.
        no_prev_churns:
            Indicator if the individual has not previously churned.
        payment_method_id:
            The payment method.

    References:
        [1] Håvard Kvamme and Ørnulf Borgan. The Brier Score under Administrative Censoring: Problems
            and Solutions. arXiv preprint arXiv:1912.08581, 2019.
            https://arxiv.org/pdf/1912.08581.pdf

        [2] Håvard Kvamme, Ørnulf Borgan, and Ida Scheel. Time-to-event prediction with neural networks
            and Cox regression. Journal of Machine Learning Research, 20(129):1–30, 2019.
            http://jmlr.org/papers/v20/18-424.html
    """
    name='kkbox'
    _checksum = 'cc69aee5f48e401e1bdf47a13dca891190257da62a12aa263f75d907b8c24240'

    def __init__(self):
        self._path_dir = _PATH_DATA / self.name
        self.path_survival = self._path_dir / 'survival_data.feather'
        self.path_covariates = self._path_dir / 'covariates.feather'
        self.log_cols = ['actual_amount_paid', 'days_between_subs', 'days_since_reg_init',
                         'payment_plan_days', 'plan_list_price']

    def read_df(self, log_trans=True, no_covs=False):
        """Get the data set as a pandas data frame.

        The columns: 'duration' and 'event' gives the duration time and event indicator, and
        the column 'censor_duration' give the administrative censoring time.

        The survival data set contains no covariates, but can be useful for extending
        the dataset with more covariates from Kaggle.
    
        Keyword Arguments:
            log_trans {bool} -- If covariates in 'kkbox.log_cols' (from Kvamme paper) should be
                transformed with 'z = log(x - min(x) + 1)'. (default: {True})
            no_covs {str} -- If False get the regular data set, if True only get the survival set
                without the covariates. (default: {False})
        """
        if no_covs is False:
            path = self.path_covariates
        elif no_covs is True:
            path = self.path_survival
            return pd.read_feather(self.path_survival)
        else:
            raise NotImplementedError

        if not path.exists():
            print(f"""
            The KKBox dataset not locally available.
            If you want to download, call 'kkbox.download_kkbox()', but note that 
            this might take around 10 min!
            NOTE: You need Kaggle credentials! Follow instructions at 
            https://github.com/Kaggle/kaggle-api#api-credentials
            """)
            return None

        def log_min_p(col, df):
            x = df[col]
            min_ = -1. if col == 'days_since_reg_init' else 0.
            return np.log(x - min_ + 1)

        df = pd.read_feather(path).set_index('index_survival')
        if log_trans:
            df = df.assign(**{col: log_min_p(col, df) for col in self.log_cols})
            df = df.rename(columns={col: f"log_{col}" for col in self.log_cols})

        drop_cols = ['duration', 'churn', 'duration_censor']
        df = df.loc[(df['duration_lcd'] > 0) & (df['duration_censor_lcd'] > 0)]
        df = df.drop(drop_cols, axis=1)
        df = df.rename(columns=dict(duration_lcd='duration', churn_lcd='event',
                                    duration_censor_lcd='censor_duration'))
        return df
    
    def _make_train_test_split(self, seed=1234):
        pass

    def _make_survival_data(self):
        """Combine the downloaded files and create a survival data sets
        (more or less without covariates).

        A customer is considered churned if one of the following is true:
            - If it has been more than 30 days since the expiration data of a membership subscription until the next transaction.
            - If the customer has expiration in before 2017-03-01, and no transaction after that.
        We include two form for administrative censoring:
            - One marked by 'lcd' (last_churn_date), where we perform administrative censoring at '2017-01-29'.
            - One where we use the 'membership_expire_date' for the churned individuals and 'last_churn_date'
              for the churned individuals.
        """
        train = pd.read_feather(self._path_dir / 'train_raw.feather')
        members = pd.read_feather(self._path_dir / 'members.feather')
        trans = (pd.read_feather(self._path_dir / 'transactions.feather')
                [['msno', 'transaction_date', 'membership_expire_date', 'is_cancel']])
        LAST_CHURN_DATE = '2017-01-29' # 30 days before last transactions are made in the dataset.

        # Churn: More than 30 days before reentering
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
                                    (x['max_trans_date'] <= LAST_CHURN_DATE) &
                                    (x['transaction_date'] == x['max_trans_date']) & 
                                    (x['membership_expire_date'] <= LAST_CHURN_DATE)
                                    )))
        # Churn: From training set
        trans = (trans
                .merge(train, how='left', on='msno')
                .assign(train_churn=lambda x: x['is_churn'].fillna(0).astype('bool'))
                .drop('is_churn', axis=1)
                .assign(train_churn=lambda x: (x['max_trans_date'] == x['transaction_date']) & x['train_churn'])
                .assign(churn=lambda x: x['train_churn'] | x['churn30'] | x['final_churn']))
        # split individuals on churn
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

        # Get only last transactions (per churn)
        indivs = (trans
                .assign(last_start=lambda x: (x['n_prev_churns'] == x.groupby('msno')['n_prev_churns'].transform('max')))
                .assign(censored=(lambda x: (x['last_start']) & (x['churn'] == False) &
                                    (x['transaction_date'] == x['max_trans_date'])))
                .loc[lambda x: x['censored'] | x['churn']]
                .merge(members[['msno', 'registration_init_time']], how='left', on='msno'))

        def time_diff_days(df, last, first):
            return (df[last] - df[first]).dt.total_seconds() / (60 * 60 * 24)

        indivs = (indivs
                .assign(duration=lambda x: time_diff_days(x, 'membership_expire_date', 'start_date'),
                        days_since_reg_init=lambda x: time_diff_days(x, 'start_date', 'registration_init_time')))
        # Add administrative censoring durations.
        # We add to types:
        # - Censoring based on the fixed data `LAST_CHURN_DATE` (lcd).
        # - Censoring based on member ship expiration, where churned 
        #   are censored at `last_churn_date`. 
        indivs = (indivs
                .assign(duration_censor=lambda x: x['duration'],
                        last_churn_date=pd.to_datetime(LAST_CHURN_DATE),
                        duration_lcd=lambda x: x['duration'],
                        churn_lcd=lambda x: x['churn'])
                .assign(duration_censor_lcd=lambda x: time_diff_days(x, 'last_churn_date', 'start_date'))
                .drop('last_churn_date', axis=1))
        indivs.loc[lambda x: x['churn'], 'duration_censor'] = indivs['duration_censor_lcd']
        indivs = indivs.assign(fix_lcd=lambda x: x['duration'] > x['duration_censor_lcd'])
        indivs.loc[lambda x: x['fix_lcd'], 'churn_lcd'] = False
        indivs.loc[lambda x: x['fix_lcd'], 'duration_lcd'] = indivs['duration_censor_lcd']
        indivs = indivs.drop('fix_lcd', axis=1)
        assert (indivs
                .assign(diff=lambda x: x['duration_censor'] - x['duration'])
                .loc[lambda x: x['diff'] < 0]
                ['train_churn'].all()), 'All strange censoring times should come from train_churn'
        # Drop the churns from 'train_churn' that does not fit with our administrative censoring
        indivs = indivs.loc[lambda x: x['duration_censor'] >= x['duration']]
        assert (indivs['duration_censor_lcd'] >= indivs['duration_lcd']).all(), 'Cannot have censor durations smaller than durations'
        tmp = indivs.loc[lambda x: x['churn'] == False]
        assert (tmp['duration'] == tmp['duration_censor']).all(), 'Need all censor durations to be equal'
        tmp = indivs.loc[lambda x: x['churn_lcd'] == False]
        assert (tmp['duration_lcd'] == tmp['duration_censor_lcd']).all(), 'Need all censor durations to be equal'

        # When multiple transactions on last day, remove all but the last
        indivs = indivs.loc[lambda x: x['transaction_date'] != x['next_trans_date']] 
        assert indivs.shape == indivs.drop_duplicates(['msno', 'transaction_date']).shape
        assert (indivs['churn'] != indivs['censored']).all()

        # Clean up and remove variables that are not from the first transaction day
        dropcols = ['transaction_date', 'is_cancel', 'next_trans_date', 'max_trans_date', 'prev_mem_exp_date',
                    'censored', 'churn30', 'final_churn', 'train_churn', 'membership_expire_date', 'last_start',
                    'new_start', 'first_churn']

        indivs = (indivs
                .assign(churn_type=lambda x: 1*x['churn30'] + 2*x['final_churn'] + 4*x['train_churn'])
                .assign(churn_type=lambda x: 
                        np.array(['censoring', '30days', 'final', '30days_and_final', 'train', 'train_and_30', 
                                    'train_and_final', 'train_30_and_final'])[x['churn_type']])
                .drop(dropcols, axis=1))

        indivs = indivs.loc[lambda x: x['churn_type'] != 'train_30_and_final']
        indivs = indivs.loc[lambda x: x['duration'] > 0]

        def as_category(df, columns):
            return df.assign(**{col: df[col].astype('category') for col in columns})

        def as_int(df, columns):
            return df.assign(**{col: df[col].round().astype('int') for col in columns})

        indivs = (indivs
                .pipe(as_int, ['duration', 'duration_censor', 'duration_lcd', 'duration_censor_lcd'])
                .pipe(as_category, ['churn_type']))
        indivs.reset_index(drop=True).to_feather(self.path_survival)

    def _make_survival_covariates(self):
        individs = pd.read_feather(self.path_survival)
        members = pd.read_feather(self._path_dir / 'members.feather')

        trans = (individs
                .merge(pd.read_feather(self._path_dir / 'transactions.feather'), 
                        how='left', left_on=['msno', 'start_date'], right_on=['msno', 'transaction_date'])
                .drop(['transaction_date'], axis=1) # same as start_date
                .drop_duplicates(['msno', 'start_date'], keep='last') # keep last transaction on start_date (by idx)
                )
        assert trans.shape[0] == individs.shape[0]

        def get_age_at_start(df):
            fixed_date = pd.datetime(2017, 3, 1)
                # Not important what the date is, though it is reasonable to use the last.
            age_diff = (fixed_date - df['start_date']).dt.total_seconds() / (60*60*24*365)
            return np.round(df['bd'] - age_diff)

        trans = (trans
                .merge(members.drop(['registration_init_time'], axis=1), how='left', on='msno')
                .assign(age_at_start=lambda x: get_age_at_start(x))
                .drop(['bd'], axis=1)
                .assign(strange_age=lambda x: (x['age_at_start'] <= 0) | (x['age_at_start'] >= 100),
                        age_at_start=lambda x: x['age_at_start'].clip(lower=0, upper=100)))
        # days_between_subs 
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
        # We have removed negative entries, so we set Nans to -1, but don't add dummy because its equal to days_since_reg_init dummy.
        trans = trans.assign(age_at_start=lambda x: x['age_at_start'].fillna(-1.))
        # We use n_prev_churns == 0 as an indicator that there are no previous churn 
        trans = (trans
                .assign(no_prev_churns=lambda x: x['n_prev_churns'] == 0))
        # Drop variables that are not useful 
        trans = (trans
                .drop(['start_date', 'registration_init_time', 'churn_type', 
                        'membership_expire_date'],
                    axis=1))

        bool_cols = ['is_auto_renew', 'is_cancel']
        cat_cols = ['payment_method_id', 'city', 'gender', 'registered_via']
        int_cols = ['days_between_subs', 'days_since_reg_init', 'age_at_start']
        trans = trans.assign(**{col: trans[col].astype('bool') for col in bool_cols}) 
        trans = trans.assign(**{col: trans[col].cat.remove_unused_categories() for col in cat_cols}) 
        trans = trans.assign(**{col: trans[col].round().astype('int') for col in int_cols}) 

        trans = trans.assign(index_survival=trans.index.values).reset_index(drop=True)
        last_cols = ['duration', 'churn', 'duration_censor', 'duration_lcd', 'churn_lcd', 'duration_censor_lcd']
        first_cols = ['index_survival', 'msno']
        trans = trans[first_cols + list(trans.columns.drop(first_cols + last_cols)) + last_cols]

        trans.to_feather(self.path_covariates)

    def _clean_up(self):
        remove = ['train.csv.7z', 'transactions.csv.7z', 'members_v3.csv.7z',
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
        for path in [self.path_covariates, self.path_survival]:
            path.unlink()
