# -*- coding: utf-8 -*-

'''Simulation used in article.
'''

import numpy as np
import pandas as pd
import scipy

class SimulationRelativeRisk(object):
    '''Abstract class for simulation relative risk survival data,
    with constant baseline, and constant censoring distribution

    Parameters:
        h0: Is baseline constant.
        right_c: Time for right censoring.
        c0: Constant censoring distribution
    '''
    def __init__(self, h0, right_c, c0=30.):
        self.h0 = h0
        self.right_c = right_c
        self.c0 = c0
    
    def simulate(self, n):
        '''Returns (x, t, d)
        x: covariates
        t: Event times
        d: Event indicator
        '''
        x = self.sample_x(n)
        v = np.random.exponential(size=n)
        t = self.inv_cum_hazard(v, x)
        c = self.c0 * np.random.exponential(size=n)
        tt = t.copy()
        tt[c < t] = c[c < t]
        tt[tt > self.right_c] = self.right_c
        d = tt == t
        return x, tt, d
    
    @staticmethod
    def sample_x(n):
        raise NotImplementedError

    def inv_cum_hazard(self, v, x):
        '''The inverse of the cumulative hazard.'''
        raise NotImplementedError
    
    def cum_hazard(self, t, x):
        '''The the cumulative hazard function.'''
        raise NotImplementedError

    def survival_func(self, t, x):
        '''Returns the survival function.'''
        return np.exp(-self.cum_hazard(t, x))
    
    def survival_grid_single(self, x, t=None):
        x = x.reshape(1, -1)
        if t is None:
            t = np.arange(0, 31, 0.5)
        return pd.Series(self.survival_func(t, x), index=t)
    
    def survival_grid(self, x, t=None):
        if t is None:
            t = np.arange(0, 31, 0.5)
        s = [self.survival_grid_single(xx, t) for xx in x]
        return pd.concat(s, axis=1)


class LinearPH(SimulationRelativeRisk):
    '''Survival simulations study for linear prop. hazard model
        h(t | x) = h0 exp[g(x)], where g(x) is linear.

    Parameters:
        h0: Is baseline constant.
        right_c: Time for right censoring.
    '''
    @staticmethod
    def sample_x(n):
        return np.random.uniform(-1, 1, size=(n, 3))

    @staticmethod
    def g(x):
        x0, x1, x2 = x[:, 0], x[:, 1], x[:, 2]
        return 0.44 * x0 + 0.66 * x1 + 0.88 * x2

    def inv_cum_hazard(self, v, x):
        '''The inverse of the cumulative hazard.'''
        return v / (self.h0 * np.exp(self.g(x)))

    def cum_hazard(self, t, x):
        '''The the cumulative hazard function.'''
        return self.h0 * t * np.exp(self.g(x))


class NonLinearPH(LinearPH):
    '''Survival simulations study for non-linear prop. hazard model
        h(t | x) = h0 exp[g(x)], where g(x) is non-linear.

    Parameters:
        h0: Is baseline constant.
        right_c: Time for right censoring.
    '''
    @staticmethod
    def g(x):
        x0, x1, x2 = x[:, 0], x[:, 1], x[:, 2]
        beta = 2/3
        linear = LinearPH.g(x)
        nonlinear =  beta * (x0**2 + x2**2 + x0*x1 + x1*x2 + x1*x2)
        return linear + nonlinear


class NonLinearNonPH(NonLinearPH):
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
    @staticmethod
    def a(x):
        _, _, x2 = x[:, 0], x[:, 1], x[:, 2]
        # return np.tanh(100*x2) + NonLinearPH.g(x)  # Maybe change to np.sign(x2)??
        return np.sign(x2) + NonLinearPH.g(x) 
    
    @staticmethod
    def b(x):
        x0, x1, _ = x[:, 0], x[:, 1], x[:, 2]
        return np.abs(0.2 * (x0 + x1) + 0.5 * x0 * x1)

    @staticmethod
    def g(t, x):
        return NonLinearNonPH.a(x) + NonLinearNonPH.b(x) * t
    
    def inv_cum_hazard(self, v, x):
        return 1 / self.b(x) * np.log(1 + v * self.b(x) / self.h0 * np.exp(-self.a(x)))
    
    def cum_hazard(self, t, x):
        return self.h0 / self.b(x) * np.exp(self.a(x)) * (np.exp(self.b(x)*t) - 1)
    