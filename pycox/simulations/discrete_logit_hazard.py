
import numpy as np
import pandas as pd
import torchtuples as tt

from pycox.simulations import base

_TIMES = np.linspace(0, 100, 1001)


class SimBase(base._SimBase):
    times = _TIMES
    num_weights = NotImplemented
    def __init__(self, covs_per_weight=5, betas=None):
        self.covs_per_weight = covs_per_weight
        self.betas = betas if betas else self.make_betas()

    def make_betas(self, func=lambda m: np.random.normal(0, 1, m)):
        return tuple(func(self.covs_per_weight) for _ in range(self.num_weights))

    @staticmethod
    def _sample_uniform(n):
        return np.random.uniform(-1, 1, (n, 1))

    def sample_weights(self, n):
        return [self._sample_uniform(n) for _ in range(self.num_weights)]

    def sample_covs(self, weights):
        return [self._conditional_covariate_sampling(beta, weight)
               for beta, weight in zip(self.betas, weights)]

    def surv_df(self, logit_haz):
        assert len(self.times) == (logit_haz.shape[1] + 1), 'Need dims to be correct'
        haz = sigmoid(logit_haz)
        surv = np.ones((len(self.times), len(haz)))
        surv[1:, :] = haz2surv(haz).transpose()
        return pd.DataFrame(surv, index=self.times)

    @staticmethod
    def _conditional_covariate_sampling(beta, weight):
        beta, weight = beta.reshape(-1), weight.reshape(-1)
        size = len(weight), len(beta)
        u = np.random.uniform(-1, 1, size=size)
        u[:, 0] = weight
        x = np.empty_like(u)
        x[:, :-1] = -np.diff(u)/beta[:-1]
        x[:, -1] = (u[:, 0] - x[:, :-1].dot(beta[:-1]))/beta[-1]
        return x

    def sample_event_times(self, logit_haz):
        haz = sigmoid(logit_haz)
        assert haz.shape[1] == len(self.times)-1, 'Fix dims'
        samp = np.random.uniform(0, 1, haz.shape)
        hit = np.zeros((len(haz), len(self.times)), 'bool')
        hit[:, 1:] = samp < haz
        idx_first = hit.argmax(1)
        durations = self.times[idx_first]  # -1 because hit has one additional column
        durations[idx_first == False] = np.nan
        return durations

    def simulate(self, n, surv_df=False):
        weights = self.sample_weights(n)
        return self.simulate_from_weights(weights, surv_df)

    def simulate_from_weights(self, weights, surv_df=False):
        logit_haz = self.logit_haz(self.times[1:], *weights)
        durations = self.sample_event_times(logit_haz)#.astype('float32')
        is_nan = np.isnan(durations)
        events = np.ones_like(durations)
        events[is_nan] = 0.
        durations[is_nan] = self.times[-1]
        covs = self.sample_covs(weights)
        covs = tt.tuplefy(covs).flatten()
        covs = np.concatenate(covs, axis=1)#.astype('float32')
        surv = self.surv_df(logit_haz) if surv_df is True else None
        return dict(covs=covs, durations=durations, events=events, weights=weights,
                    surv_df=surv)


    def covs2weights(self, covs):
        return [cov.dot(beta).reshape(-1, 1) for cov, beta in zip(covs, self.betas)]

    def covs2surv_df(self, covs):
        weights = self.covs2weights(covs)
        logit_haz = self.logit_haz(self.times[1:], *weights)
        return self.surv_df(logit_haz)

    def logit_haz(self, times, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def dict2df(data):
        """Make a pd.DataFrame from the dict obtained when simulating.

        Arguments:
            data {dict} -- Dict from simulation.

        Returns:
            pd.DataFrame -- A DataFrame
        """
        return base.dict2df(data, False)


class SimSin(SimBase):
    num_weights = 4
    def logit_haz(self, times, a, bb, c, dd):
        """We expect a, bb, c, dd to be Unif[-1, 1] and transform them to
        the desired ranges. Use '_logit_haz' to skip this transform.
        """
        a = a * 5  # Unif[-5, 5]
        idx = ((bb + 1) / 2 * 5).astype('int')
        bb = np.arange(-1, 4)[idx]  # Unif[{-1, 0, 1, 2, 3}]
        c = c * 15  # Unif[-15, 15]
        dd = dd * 2  # Unif[-2, 2]
        return self._logit_haz(times, a, bb, c, dd)

    @staticmethod
    def _logit_haz(times, a, bb, c, dd):
        b = 2 * np.pi / 100 * np.power(2., bb)
        d = dd - 6 - abs(a/2)
        return a * np.sin(b*(times + c)) + d


class SimConstHaz(SimBase):
    num_weights = 1
    def logit_haz(self, times, a):
        """Expect a to be Unit[-1, 1]."""
        a = (a + 1) / 2 * 5 - 8  # Unif[-8, -3]
        return self._logit_haz(times, a)

    @staticmethod
    def _logit_haz(times, a):
        return a * np.ones((len(a), len(times)))


class SimAcceleratingHaz(SimBase):
    num_weights = 1
    def logit_haz(self, times, aa):
        """Expect a to be Unit[-1, 1]."""
        aa = (aa + 1) / 2 * 6 - 5  # Unif[-5, 1]
        a = sigmoid(aa)
        return self._logit_haz(times, a)

    @staticmethod
    def _logit_haz(times, a):
        start = -10
        return a * times + start


class SimConstHazIndependentOfWeights(SimBase):
    """Constant hazards independen of weights and covariates.
    Covariates are simply a column of zeros and can be removed from analysis.
    
    None of the call arguments matter, as they are set in the constructor.
    """
    num_weights = 1
    def __init__(self, *args, **kwargs):
        covs_per_weight = 1
        betas = np.array([0.])
        super().__init__(covs_per_weight, betas)

    def sample_weights(self, n):
        return [np.zeros((n, 1))]

    def sample_covs(self, weights):
        return weights

    def covs2weights(self, covs):
        return covs

    def logit_haz(self, times, a):
        return -7. * np.ones((len(a), len(times)))


class SimUniform(SimBase):
    num_weights = 1
    def __init__(self, s_end=0.2, *args, **kwargs):
        self.s_end = s_end
        covs_per_weight = 1
        betas = np.array([0.])
        super().__init__(covs_per_weight, betas)

    def logit_haz(self, times, w):
        n, m = len(w), len(times)
        j = np.arange(1, m+1, dtype='float').reshape(1, -1).repeat(n, axis=0)
        return -np.log(m/(1-self.s_end) - j + 1)

    def sample_weights(self, n):
        return [np.zeros((n, 1))]

    def sample_covs(self, weights):
        return weights

    def covs2weights(self, covs):
        return covs


class _SimCombine(SimBase):
    sims = NotImplemented
    alpha_range = NotImplemented
    _first_prev = NotImplemented

    def sample_weights(self, n):
        weights = [sim.sample_weights(n) for sim in self.sims]
        return [super().sample_weights(n)] + weights

    def sample_covs(self, weights):
        alpha = weights[0]
        covs = [sim.sample_covs(w) for sim, w in zip(self.sims, weights[1:])]
        return [super().sample_covs(alpha)] + covs

    def logit_haz(self, times, *weights):
        alpha = np.concatenate(weights[0], axis=1)
        alpha[:, 0] += self._first_pref
        alpha = softmax(alpha * self.alpha_range)
        logit_haz = 0.
        for i, (sim, w) in enumerate(zip(self.sims, weights[1:])):
            logit_haz += sim.logit_haz(self.times[1:], *w) * alpha[:, [i]]
        return logit_haz

    def covs2weights(self, covs):
        weights = [sim.covs2weights(cov) for sim, cov in zip(self.sims, covs[1:])]
        return [super().covs2weights(covs[0])] + weights

    def covs2surv_df(self, covs):
        weights = self.covs2weights(covs)
        logit_haz = self.logit_haz(self.times[1:], *weights)
        return self.surv_df(logit_haz)


class SimSinAccConst(_SimCombine):
    def __init__(self, covs_per_weight=5, alpha_range=5., sin_pref=0.6):
        self.num_weights = 3
        super().__init__(covs_per_weight)
        self.alpha_range = alpha_range
        self._first_pref = sin_pref
        self.sim_sin = SimSin(covs_per_weight)
        self.sim_const = SimConstHaz(covs_per_weight)
        self.sim_acc = SimAcceleratingHaz(covs_per_weight)
        self.sims = [self.sim_sin, self.sim_const, self.sim_acc]


class SimConstAcc(_SimCombine):
    def __init__(self, covs_per_weight=5, alpha_range=5., const_pref=2):
        self.num_weights = 2
        super().__init__(covs_per_weight)
        self.alpha_range = alpha_range
        self._first_pref = const_pref
        self.sim_const = SimConstHaz(covs_per_weight)
        self.sim_acc = SimAcceleratingHaz(covs_per_weight)
        self.sims = [self.sim_const, self.sim_acc]


class _SimStudyBase:
    sim_surv = NotImplemented
    sim_censor = NotImplemented
    @staticmethod
    def _combine_surv_and_censor(surv, censor):
        surv['durations_true'], surv['events_true'] = surv['durations'].copy(), surv['events'].copy()
        is_censor = censor['durations'] < surv['durations']
        surv['durations'][is_censor] = censor['durations'][is_censor]
        surv['events'][is_censor] = 0.
        return dict(**surv, **{'censor_'+str(k): v for k, v in censor.items()})

    def simulate(self, n, surv_df=False, censor_df=False, binary_surv=False):
        if binary_surv:
            if not (surv_df and censor_df):
                raise ValueError("To produce binary_surv, you need to also set surv_df and censor_df to True")
        surv = self.sim_surv.simulate(n, surv_df)
        censor = self.sim_censor.simulate(n, censor_df)
        res = self._combine_surv_and_censor(surv, censor)
        if binary_surv:
            res['binary_surv_df'] = self.binary_surv(res)
        return res

    @staticmethod
    def dict2df(data, add_true=True, add_censor_covs=False):
        """Make a pd.DataFrame from the dict obtained when simulating.

        Arguments:
            data {dict} -- Dict from simulation.

        Keyword Arguments:
            add_true {bool} -- If we should include the true duration and censoring times
                (default: {True})
            add_censor_covs {bool} -- If we should include the censor covariates as covariates.
                (default: {False})

        Returns:
            pd.DataFrame -- A DataFrame
        """
        return base.dict2df(data, add_true, add_censor_covs)


class SimStudyIndepSurvAndCens(_SimStudyBase):
    def __init__(self, sim_surv, sim_censor):
        self.sim_surv = sim_surv
        self.sim_censor = sim_censor


class SimStudySACCensorConst(_SimStudyBase):
    def __init__(self, covs_per_weight=5, alpha_range=5., sin_pref=0.6):
        """Simulation study from [paper link].
        It combines three sources to the logit-hazard: A sin function, increasing function
        and a constant function.

        See paper for details.
        
        Keyword Arguments:
            covs_per_weight {int} -- Number of covariates per weight (gamma in paper)
                 (default: {5})
            alpha_range {[type]} -- Controls how the mixing between the three logit-hazards.
                High alpha is equivalent to picking one of them, while low is equivalent to
                a more homogeneous mixing. (default: {5.})
            sin_pref {float} -- Preference for the SimSin in the mixing. (default: {0.6})
        """
        self.sim_surv = SimSinAccConst(covs_per_weight, alpha_range, sin_pref)
        self.sim_censor = SimConstHazIndependentOfWeights()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def haz2surv(haz, eps=1e-7):
    return np.exp(np.log((1 - haz) + eps).cumsum(1))

def softmax(x):
    exp = np.exp(x)
    return exp / exp.sum(1, keepdims=True)