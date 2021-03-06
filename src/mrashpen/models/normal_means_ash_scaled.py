"""
Class for normal means model
with ASH prior
    p(y | b, s^2) = N(y | mu, s^2 / d)
    p(mu) = sum_k w_k N(mu | 0, s^2 * s_k^2)
y, mu, d: array of size N
s: single variable
w_k, s_k: array of size K
"""

import numpy as np
import functools
import random
import logging

from ..utils.logs import MyLogger
from ..utils.decorators import run_once

class NormalMeansASHScaled:

    def __init__(self, y, s, wk, sk, d = 1.0, debug = True):
        '''
        y, d are vectors of length N
        s is a single variable
        wk, sk are vectors fo length K 
        wk are prior mixture proportions and sk are prior mixture variances
        '''
        self._y  = y
        self._wk = wk
        self._sk = sk
        self._k  = wk.shape[0]
        self._n  = y.shape[0]
        self._s  = s
        self._s2 = s * s
        self._d  = d
        self._nonzero_widx = np.where(wk != 0)[0]
        if not isinstance(d, np.ndarray):
            self._d = np.repeat(d, self._n)
        if debug:
            self.logger = MyLogger(__name__)
        else:
            self.logger = MyLogger(__name__, level = logging.INFO)
        self.randomseed = random.random()


    def __hash__(self):
        return hash(self.randomseed)


    def set_s2_eps(self, eps):
        '''
        This is only used for checking derivatives.
        Adds a small value, eps to s2 for calculating derivatives numerically.
        '''
        self._s2 += eps
        self._s = np.sqrt(self._s2)
        return
        

    @property
    def y(self):
        return self._y


    @property
    def yvar(self):
        return self._s2 / self._d


    '''
    log(sum(exp(z))) = M + log(sum(exp(z - M)))
    '''
    def log_sum_exponent(self, z):
        zmax = np.max(z, axis = 1)
        logsum = np.log(np.sum(np.exp(z-zmax.reshape(-1, 1)), axis = 1)) + zmax
        return logsum


    '''
    log(sum(wk * Ljk))
    takes care of zero values in wk
    requires log(Ljk) as input
    '''
    def log_sum_wkLjk(self, logLjk):
        z = logLjk[:, self._nonzero_widx] + np.log(self._wk[self._nonzero_widx])
        return self.log_sum_exponent(z)


    '''
    log(sum(wk * fjk * Ljk))
    takes care of negative values in fk
    requires fk and log(Ljk) as input
    '''
    def log_sum_wfL(self, fjk, logLjk):
        logwk  = np.log(self._wk[self._nonzero_widx])
        logfjk = np.log(np.abs(fjk[:, self._nonzero_widx]))
        z = logwk + logfjk + logLjk[:, self._nonzero_widx]
        zmax = np.max(z, axis = 1)
        ezk = np.exp(z - zmax.reshape(-1, 1))
        logsum = zmax + np.log(np.sum(ezk * np.sign(fjk)[:, self._nonzero_widx], axis = 1))
        return logsum


    def logLjk(self, derive = 0):
        self.calculate_logLjk()
        return self._logLjk[derive]


    @run_once
    def calculate_logLjk(self, derive = 0):
        '''
        this is one part of the posterior in normal means model. LogLjk is defined as:    
            p(y | f, s2)   =   (1 / sqrt(2 * pi)) * sum_k [w_k * exp(logLjk)]            # derive = 0
            p'(y | f, s2)  = - (y / sqrt(2 * pi)) * sum_k [w_k * exp(logLjk)]            # derive = 1 (first derivative)
            p''(y | f, s2) = (y^2 / sqrt(2 * pi)) * sum_k [w_k * exp(logLjk)] + p' / y   # derive = 2 
        returns N x K matrix
        '''
        #self.logger.debug(f"Calculating logLjk for NM model hash {self.__hash__()}")
        s2  = self._s2
        sk2 = np.square(self._sk).reshape(1, self._k)
        y2  = np.square(self._y).reshape(self._n, 1)
        dj  = self._d.reshape(self._n, 1)
        v2  = sk2 + (1 / dj)
        logs2 = np.log(s2)
        logv2 = np.log(v2)
        y2_over_v2s2 = y2 / (v2 * s2)
        # N x K length vector of posterior variances
        # v2 = s2 + sk2
        v2 = sk2 + (1 / dj)
        self._logLjk = {}
        self._logLjk[0] = - 0.5 * (     logs2 + logv2  + y2_over_v2s2) # N x K matrix
        self._logLjk[1] = - 0.5 * (3 * (logs2 + logv2) + y2_over_v2s2) # N x K matrix
        self._logLjk[2] = - 0.5 * (5 * (logs2 + logv2) + y2_over_v2s2) # N x K matrix
        return


    @property
    def logML(self):
        self.calculate_logML()
        return self._logML


    @run_once
    def calculate_logML(self):
        self._logML = - 0.5 * np.log(2 * np.pi) + self.log_sum_wkLjk(self.logLjk())
        return


    '''
    ML_deriv / (ML * y)
    '''
    @property
    def ML_deriv_over_ML_y(self):
        self.calculate_ML_deriv_over_ML_y()
        return self._ML_deriv_over_ML_y


    @run_once
    def calculate_ML_deriv_over_ML_y(self):
        log_numerator   = self.log_sum_wkLjk(self.logLjk(derive = 1))
        log_denominator = self.log_sum_wkLjk(self.logLjk())
        self._ML_deriv_over_ML_y = - np.exp(log_numerator - log_denominator)
        return


    '''
    lNM' = ML_deriv / ML
    '''
    @property
    def logML_deriv(self):
        self.calculate_logML_deriv()
        return self._logML_deriv


    @run_once
    def calculate_logML_deriv(self):
        self._logML_deriv = self.ML_deriv_over_ML_y * self._y
        return


    '''
    ML_deriv2 / ML
    '''
    @property
    def ML_deriv2_over_ML(self):
        self.calculate_ML_deriv2_over_ML()
        return self._ML_deriv2_over_ML


    @run_once
    def calculate_ML_deriv2_over_ML(self):
        log_numerator   = self.log_sum_wkLjk(self.logLjk(derive = 2))
        log_denominator = self.log_sum_wkLjk(self.logLjk())
        self._ML_deriv2_over_ML = self.ML_deriv_over_ML_y \
                                    + self._y * self._y * np.exp(log_numerator - log_denominator)
        return 


    '''
    lNM'' = (ML_deriv2 / ML) - (lNM')^2
    '''
    @property
    def logML_deriv2(self):
        self.calculate_logML_deriv2()
        return self._logML_deriv2


    @run_once
    def calculate_logML_deriv2(self):
        self._logML_deriv2 = self.ML_deriv2_over_ML - np.square(self.logML_deriv)
        return


    '''
    d lNM / dw = Ljk(0) / (ML * sqrt(2 pi))
    log (d lNM / dw) = - 0.5 log(2 pi) + log(Ljk(0)) - logML
    '''
    @property
    def logML_wderiv(self):
        self.calculate_logML_wderiv()
        return self._logML_wderiv


    @run_once
    def calculate_logML_wderiv(self):
        self._logML_wderiv = np.exp(- 0.5 * np.log(2 * np.pi) + self.logLjk() - self.logML.reshape(self._n, 1))
        return


    '''
    d lNM' / dw (eq. 65)
    '''
    @property
    def logML_deriv_wderiv(self):
        self.calculate_logML_deriv_wderiv()
        return self._logML_deriv_wderiv
        

    @run_once
    def calculate_logML_deriv_wderiv(self):
        Ljk1_over_Ljk0 = np.exp(self.logLjk(derive = 1) - self.logLjk())
        self._logML_deriv_wderiv = - self.logML_wderiv \
            * (self._y.reshape(-1, 1) * Ljk1_over_Ljk0  + self.logML_deriv.reshape(self._n,1))
        return


    '''
    d lNM / d(sigma^2)
    '''
    @property
    def logML_s2deriv(self):
        self.calculate_logML_s2deriv()
        return self._logML_s2deriv


    @run_once
    def calculate_logML_s2deriv(self):
        y2  = np.square(self._y)
        s2  = self._s2
        dj  = self._d.reshape(self._n, 1)
        sk2 = np.square(self._sk).reshape(1, self._k)
        #zjk = 0.5 * (sk2 + (1 / dj) - (y2 / s2))
        #log_numerator   = self.log_sum_wkLjk(self.logLjk(derive = 1) + zjk)
        #log_numerator   = self.log_sum_wfL(zjk, self.logLjk(derive = 1))
        log_numerator1  = self.log_sum_wkLjk(self.logLjk(derive = 1))
        log_numerator2  = self.log_sum_wkLjk(self.logLjk(derive = 1) + np.log(sk2 + 1 / dj))
        log_denominator = self.log_sum_wkLjk(self.logLjk())
        self._logML_s2deriv = 0.5 * (y2 / s2) * np.exp(log_numerator1 - log_denominator) \
                                        - 0.5 * np.exp(log_numerator2 - log_denominator)
        #self._logML_s2deriv = - np.exp(log_numerator - log_denominator)
        return


    '''
    d lNM' / d(sigma^2)
    '''
    @property
    def logML_deriv_s2deriv(self):
        self.calculate_logML_deriv_s2deriv()
        return self._logML_deriv_s2deriv


    @run_once
    def calculate_logML_deriv_s2deriv(self):
        y2  = np.square(self._y)
        s2  = self._s2
        dj  = self._d.reshape(self._n, 1)
        sk2 = np.square(self._sk).reshape(1, self._k)
        #zjk = 0.5 * (3 * sk2 + (3 / dj) - (y2 / s2))
        log_numerator1  = self.log_sum_wkLjk(self.logLjk(derive = 2))
        log_numerator2  = self.log_sum_wkLjk(self.logLjk(derive = 2) + np.log(sk2 + 1 / dj))
        #log_numerator   = self.log_sum_wfL(zjk, self.logLjk(derive = 2))
        log_denominator = self.log_sum_wkLjk(self.logLjk())
        ML_deriv_s2deriv_over_ML_y = - 0.5 * (y2 / s2) * np.exp(log_numerator1 - log_denominator) \
                                           + 1.5 * np.exp(log_numerator2 - log_denominator)
        #ML_deriv_s2deriv_over_ML = self._y * np.exp(log_numerator - log_denominator)
        self._logML_deriv_s2deriv = self._y * ML_deriv_s2deriv_over_ML_y - self.logML_deriv * self.logML_s2deriv
        return


    def posterior(self):
        self.logger.debug(f"Calculating posterior for NM model.")
        s2  = self._s2
        sk2 = np.square(self._sk).reshape(1, self._k)
        y2  = np.square(self._y).reshape(self._n, 1)
        dj  = self._d.reshape(self._n, 1)
        #v2jk  = s2 + sk2
        v2jk  = sk2 + (1 / dj)
        varjk = (s2 * sk2) / (v2jk * dj)
        mujk  = self._y.reshape(self._n, 1) * sk2 / v2jk

        phijk  = np.zeros((self._n, self._k))
        logLjk = self.logLjk()
        phijk[:, self._nonzero_widx] = logLjk[:, self._nonzero_widx] + np.log(self._wk[self._nonzero_widx])
        zjk    = logLjk[:, self._nonzero_widx] + np.log(self._wk[self._nonzero_widx])
        zjkmax = np.max(zjk, axis = 1)
        phijk[:, self._nonzero_widx] = np.exp(zjk - zjkmax.reshape(-1, 1))
        phijk /= np.sum(phijk, axis = 1).reshape(self._n, 1)
        #phijk[:, self._nonzero_widx] = np.exp(zjk - self.log_sum_wkLjk(logLjk).reshape(-1,1))
        return phijk, mujk, varjk
