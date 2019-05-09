import numpy as np
import pandas as pd
from torchtuples.tupletree import docstring


def dict2df(data, drop_true=False):
    """Make a pd.DataFrame from the dict obtained when simulating.
    
    Arguments:
        data {dict} -- Dict from simulation.
    
    Keyword Arguments:
        drop_true {bool} -- If we should include the true duration and censoring times
            (default: {False})
    
    Returns:
        pd.DataFrame -- A DataFrame
    """
    df = (pd.DataFrame(data['covs'], columns=[f"x{i}" for i in range(data['covs'].shape[1])])
          .assign(duration=data['durations'].astype('float32'),
                  event=data['events'].astype('float32')))
    if not drop_true:
        df = df.assign(duration_true=data['durations_true'].astype('float32'),
                       censoring_true=data['censor_durations'].astype('float32'))
    return df


class _SimRelativeRisk:
    '''Abstract class for simulation relative risk survival data,
    with constant baseline, and constant censoring distribution

    Parameters:
        h0: Is baseline constant.
        right_c: Time for right censoring.
        c0: Constant censoring distribution
    '''
    def __init__(self, h0, right_c=30., c0=30., surv_grid=None):
        self.h0 = h0
        self.right_c = right_c
        self.c0 = c0
        self.surv_grid = surv_grid
    
    def simulate(self, n, surv_df=False, censor_df=False):
        covs = self.sample_covs(n).astype('float32')
        v = np.random.exponential(size=n)
        t = self.inv_cum_hazard(v, covs)
        c = self.c0 * np.random.exponential(size=n)
        tt = t.copy()
        tt[c < t] = c[c < t]
        tt[tt > self.right_c] = self.right_c
        d = tt == t
        surv_df = self.surv_df(covs, self.surv_grid) if surv_df else None
        censor_surv_df = NotImplemented if censor_df else None
        return dict(covs=covs, durations=tt, events=d, surv_df=surv_df, durations_true=t,
                    events_true=np.ones_like(t), censor_durations=c,
                    censor_events=np.ones_like(c), censor_surv_df=None)

    @staticmethod
    def sample_covs(n):
        raise NotImplementedError

    def inv_cum_hazard(self, v, covs):
        '''The inverse of the cumulative hazard.'''
        raise NotImplementedError
    
    def cum_hazard(self, t, covs):
        '''The the cumulative hazard function.'''
        raise NotImplementedError

    def survival_func(self, t, covs):
        '''Returns the survival function.'''
        return np.exp(-self.cum_hazard(t, covs))
    
    def survival_grid_single(self, covs, t=None):
        covs = covs.reshape(1, -1)
        if t is None:
            t = np.arange(0, 31, 0.5)
        return pd.Series(self.survival_func(t, covs), index=t)
    
    def surv_df(self, covs, t=None):
        if t is None:
            t = np.linspace(0, 30, 100)
        s = [self.survival_grid_single(xx, t) for xx in covs]
        return pd.concat(s, axis=1)

    @staticmethod
    @docstring(dict2df)
    def dict2df(data, drop_true=False):
        return dict2df(data, drop_true)


class SimLinearPH(_SimRelativeRisk):
    '''Survival simulations study for linear prop. hazard model
        h(t | x) = h0 exp[g(x)], where g(x) is linear.

    Parameters:
        h0: Is baseline constant.
        right_c: Time for right censoring.
    '''
    def __init__(self, h0=0.1, right_c=30., c0=30., surv_grid=None):
        super().__init__(h0, right_c, c0, surv_grid)

    @staticmethod
    def sample_covs(n):
        return np.random.uniform(-1, 1, size=(n, 3))

    @staticmethod
    def g(covs):
        x = covs
        x0, x1, x2 = x[:, 0], x[:, 1], x[:, 2]
        return 0.44 * x0 + 0.66 * x1 + 0.88 * x2

    def inv_cum_hazard(self, v, covs):
        '''The inverse of the cumulative hazard.'''
        return v / (self.h0 * np.exp(self.g(covs)))

    def cum_hazard(self, t, covs):
        '''The the cumulative hazard function.'''
        return self.h0 * t * np.exp(self.g(covs))


class SimNonLinearPH(SimLinearPH):
    '''Survival simulations study for non-linear prop. hazard model
        h(t | x) = h0 exp[g(x)], where g(x) is non-linear.

    Parameters:
        h0: Is baseline constant.
        right_c: Time for right censoring.
    '''
    @staticmethod
    def g(covs):
        x = covs
        x0, x1, x2 = x[:, 0], x[:, 1], x[:, 2]
        beta = 2/3
        linear = SimLinearPH.g(x)
        nonlinear =  beta * (x0**2 + x2**2 + x0*x1 + x1*x2 + x1*x2)
        return linear + nonlinear


class SimNonLinearNonPH(SimNonLinearPH):
    '''Survival simulations study for non-linear non-prop. hazard model.
        h(t | x) = h0 * exp[g(t, x)], 
        with constant h_0, and g(t, x) = a(x) + b(x)*t.

        Cumulative hazard:
        H(t | x) = h0 / b(x) * exp[a(x)] * (exp[b(x) * t] - 1)
        Inverse:
        H^{-1}(v, x) = 1/b(x) log{1 +  v * b(x) / h0 exp[-a(x)]}

    Parameters:
        h0: Is baseline constant.
        right_c: Time for right censoring.
    '''
    def __init__(self, h0=0.02, right_c=30., c0=30., surv_grid=None):
        super().__init__(h0, right_c, c0, surv_grid)

    @staticmethod
    def a(x):
        _, _, x2 = x[:, 0], x[:, 1], x[:, 2]
        return np.sign(x2) + SimNonLinearPH.g(x) 
    
    @staticmethod
    def b(x):
        x0, x1, _ = x[:, 0], x[:, 1], x[:, 2]
        return np.abs(0.2 * (x0 + x1) + 0.5 * x0 * x1)

    @staticmethod
    def g(t, covs):
        x = covs
        return SimNonLinearNonPH.a(x) + SimNonLinearNonPH.b(x) * t
    
    def inv_cum_hazard(self, v, covs):
        x = covs
        return 1 / self.b(x) * np.log(1 + v * self.b(x) / self.h0 * np.exp(-self.a(x)))
    
    def cum_hazard(self, t, covs):
        x = covs
        return self.h0 / self.b(x) * np.exp(self.a(x)) * (np.exp(self.b(x)*t) - 1)
