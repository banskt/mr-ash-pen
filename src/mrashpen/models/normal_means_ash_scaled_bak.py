"""
Class for normal means model
with ASH prior
    p(y | b, s^2) = N(y | b, s^2)
    p(b) = sum_k w_k N(b | 0, s_k^2)
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
        y, s are vectors of length N
        wk, sk are vectors fo length K 
        wk are prior mixture proportions and sk are prior mixture variances
        '''
        self._y = y
        self._wk = wk
        self._sk = sk
        self._k = wk.shape[0]
        self._n = y.shape[0]
        self._s = s
        self._d = d
        if not isinstance(s, np.ndarray):
            self._s = np.repeat(s, self._n)
        self._s2 = np.square(self._s)
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


    @property
    def ML(self):
        '''
        marginal likelihood under mixture prior
        p(y | f, s^2) = sum_k wk * N(y | 0, s^2 + s_k^2)
        '''
        self.calculate_ML()
        return self._ML


    @run_once
    def calculate_ML(self):
        self._ML = np.dot(np.exp(self.logLjk()), self._wk) * np.sqrt(0.5 / np.pi)
        return


    @property
    def ML_deriv_over_y(self):
        '''
        returns ML_deriv(y) / y 
        ok even if y = 0
        '''
        self.calculate_ML_deriv_over_y()
        return self._ML_deriv_over_y


    @run_once
    def calculate_ML_deriv_over_y(self):
        L = np.exp(self.logLjk(derive = 1)) # N x K
        self._ML_deriv_over_y = - np.sqrt(0.5 / np.pi) * np.dot(L, self._wk)
        return


    @property
    def ML_deriv(self):
        self.calculate_ML_deriv()
        return self._ML_deriv


    @run_once
    def calculate_ML_deriv(self):
        self._ML_deriv = self.ML_deriv_over_y * self._y
        return


    @property
    def ML_deriv2(self):
        self.calculate_ML_deriv2()
        return self._ML_deriv2


    @run_once
    def calculate_ML_deriv2(self):
        L = np.exp(self.logLjk(derive = 2)) # N x K
        self._ML_deriv2 = np.sqrt(0.5 / np.pi) * np.dot(L, self._wk) * np.square(self._y) + self.ML_deriv_over_y
        return


    @property
    def ML_deriv_over_ML(self):
        '''
        ML_deriv / ML
        trying to avoid division by zero
        '''
        self.calculate_ML_deriv_over_ML()
        return self.ML_deriv_over_ML_y * self._y


    @run_once
    def calculate_ML_deriv_over_ML(self):
        self._ML_deriv_over_ML = self.ML_deriv_over_ML_y * self._y
        return


    @property
    def ML_deriv_over_ML_y(self):
        '''
        ML_deriv / (ML * y)
        trying to avoid division by zero
        '''
        self.calculate_ML_deriv_over_ML_y()
        return self._ML_deriv_over_ML_y


    @run_once
    def calculate_ML_deriv_over_ML_y(self):
        f0 = self.logLjk()
        f1 = self.logLjk(derive = 1)
        M = np.max(f0, axis = 1)
        part_numerator   = np.dot(np.exp(f1 - M.reshape(-1,1)), self._wk)
        part_denominator = np.dot(np.exp(f0 - M.reshape(-1,1)), self._wk)
        self._ML_deriv_over_ML_y = - part_numerator / part_denominator
        return


    @property
    def ML_deriv2_over_ML(self):
        '''
        ML_deriv2 / ML
        trying to avoid division by zero
        '''
        self.calculate_ML_deriv2_over_ML()
        return self._ML_deriv2_over_ML


    @run_once
    def calculate_ML_deriv2_over_ML(self):
        f0 = self.logLjk()
        f2 = self.logLjk(derive = 2)
        M = np.max(f0, axis = 1)
        part_numerator   = np.dot(np.exp(f2 - M.reshape(-1,1)), self._wk)
        part_denominator = np.dot(np.exp(f0 - M.reshape(-1,1)), self._wk)
        self._ML_deriv2_over_ML = self.ML_deriv_over_ML_y + self._y * self._y * (part_numerator / part_denominator)
        return 


    @property
    def ML_s2deriv(self):
        self.calculate_ML_s2deriv()
        return self._ML_s2deriv


    @run_once
    def calculate_ML_s2deriv(self):
        L = np.exp(self.logLjk(derive = 1)) # N x K
        s2  = self._s2.reshape(self._n, 1)
        sk2 = np.square(self._sk).reshape(1, self._k)
        y2  = np.square(self._y).reshape(self._n, 1)
        dj  = self._d.reshape(self._n, 1)
        # t2 = 1 - (y2 / (s2 + sk2)) # N x K
        t2 = (sk2 + (1 / dj)) - (y2 / s2)
        self._ML_s2deriv = - np.dot(np.multiply(L, t2), self._wk) * np.sqrt(0.125 / np.pi)
        return


    @property
    def ML_deriv_s2deriv(self):
        self.calculate_ML_deriv_s2deriv()
        return self._ML_deriv_s2deriv


    @run_once
    def calculate_ML_deriv_s2deriv(self):
        L = np.exp(self.logLjk(derive = 2)) # N x K
        s2  = self._s2.reshape(self._n, 1)
        sk2 = np.square(self._sk).reshape(1, self._k)
        y2  = np.square(self._y).reshape(self._n, 1)
        dj  = self._d.reshape(self._n, 1)
        #t2 = 3 - (y2 / (s2 + sk2)) # N x K
        t2  = 3 * (sk2 + (1 / dj)) - (y2 / s2)
        self._ML_deriv_s2deriv = np.dot(np.multiply(L, t2), self._wk) * np.sqrt(0.125 / np.pi) * self._y
        return


    @property
    def ML_s2deriv_over_ML(self):
        '''
        ML_s2deriv / ML 
        trying to avoid division by zero.
        '''
        self.calculate_ML_s2deriv_over_ML()
        return self._ML_s2deriv_over_ML


    @run_once
    def calculate_ML_s2deriv_over_ML(self):
        s2  = self._s2.reshape(self._n, 1)
        sk2 = np.square(self._sk).reshape(1, self._k)
        y2  = np.square(self._y).reshape(self._n, 1)
        dj  = self._d.reshape(self._n, 1)
        #t2 = 1 - (y2 / (s2 + sk2)) # N x K
        t2 = (sk2 + (1 / dj)) - (y2 / s2)
        f0 = self.logLjk()
        f1 = self.logLjk(derive = 1)
        M  = np.max(f0, axis = 1)
        part_numerator   = np.dot(np.exp(f1 - M.reshape(-1,1)) * t2, self._wk)
        part_denominator = np.dot(np.exp(f0 - M.reshape(-1,1)), self._wk)
        self._ML_s2deriv_over_ML = - 0.5 * (part_numerator / part_denominator)
        return 


    @property
    def ML_deriv_s2deriv_over_ML(self):
        self.calculate_ML_deriv_s2deriv_over_ML()
        return self._ML_deriv_s2deriv_over_ML


    @run_once
    def calculate_ML_deriv_s2deriv_over_ML(self):
        s2  = self._s2.reshape(self._n, 1)
        sk2 = np.square(self._sk).reshape(1, self._k)
        y2  = np.square(self._y).reshape(self._n, 1)
        dj  = self._d.reshape(self._n, 1)
        #t2 = 3 - (y2 / (s2 + sk2)) # N x K
        t2 = 3 * (sk2 + (1 / dj)) - (y2 / s2)
        f0 = self.logLjk()
        f2 = self.logLjk(derive = 2)
        M  = np.max(f0, axis = 1)
        part_numerator   = np.dot(np.exp(f2 - M.reshape(-1,1)) * t2, self._wk)
        part_denominator = np.dot(np.exp(f0 - M.reshape(-1,1)), self._wk)
        self._ML_deriv_s2deriv_over_ML = 0.5 * (part_numerator / part_denominator) * self._y
        return

        
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
        s2  = self._s2.reshape(self._n, 1)
        sk2 = np.square(self._sk).reshape(1, self._k)
        y2  = np.square(self._y).reshape(self._n, 1)
        dj  = self._d.reshape(self._n, 1)
        # N x K length vector of posterior variances
        # v2 = s2 + sk2
        v2 = s2 * (sk2 + (1 / dj))
        self._logLjk = {}
        self._logLjk[0] = -0.5 * (np.log(v2) + y2 / v2)       # N x K matrix
        self._logLjk[1] = -0.5 * (3 * np.log(v2) + (y2 / v2)) # N x K matrix
        self._logLjk[2] = -0.5 * (5 * np.log(v2) + (y2 / v2)) # N x K matrix
        return


    def posterior(self):
        self.logger.debug(f"Calculating posterior for NM model.")
        s2  = self._s2.reshape(self._n, 1)
        sk2 = np.square(self._sk).reshape(1, self._k)
        y2  = np.square(self._y).reshape(self._n, 1)
        dj  = self._d.reshape(self._n, 1)
        #v2jk  = s2 + sk2
        v2jk  = sk2 + (1 / dj)
        mujk  = self._y.reshape(self._n, 1) * sk2 / v2jk
        varjk = s2 * sk2 / v2jk / dj

        logLjk = -0.5 * (np.log(s2) + np.log(v2jk) + y2 / (s2 * v2jk))
        #phijk  = np.sqrt(0.5 / np.pi) * self._wk * np.exp(logLjk)
        phijk  = np.exp(logLjk - np.max(logLjk, axis = 1).reshape(-1, 1)) * self._wk
        phijk /= np.sum(phijk, axis = 1).reshape(self._n, 1)
        return phijk, mujk, varjk


    @property
    def logML(self):
        self.calculate_logML()
        return self._logML


    @run_once
    def calculate_logML(self):
        #self._logML = np.log(self.ML)
        f = self.logLjk()
        M = np.max(f, axis = 1)
        partML = np.dot(np.exp(f - M.reshape(-1,1)), self._wk) # to prevent overflow in np.exp(f)
        self._logML = M + np.log(partML)
        self._logML += - 0.5 * np.log( 2 * np.pi)
        return


    @property
    def logML_deriv(self):
        self.calculate_logML_deriv()
        return self._logML_deriv


    @run_once
    def calculate_logML_deriv(self):
        #self._logML_deriv = self.ML_deriv / self.ML
        self._logML_deriv = self.ML_deriv_over_ML
        return


    @property
    def logML_deriv2(self):
        self.calculate_logML_deriv2()
        return self._logML_deriv2


    @run_once
    def calculate_logML_deriv2(self):
        self._logML_deriv2 = self.ML_deriv2_over_ML - np.square(self.ML_deriv_over_ML)
        return


    @property
    def logML_wderiv(self):
        self.calculate_logML_wderiv()
        return self._logML_wderiv


    @run_once
    def calculate_logML_wderiv(self):
        # self._logML_wderiv = np.sqrt(0.5 / np.pi) * np.exp(self.logLjk()) / self.ML.reshape(self._n, 1)
        f = self.logLjk()
        M = np.max(f, axis = 1).reshape(-1, 1)
        self._logML_wderiv = np.exp(f - M) / np.dot(np.exp(f - M), self._wk).reshape(self._n, 1)
        return


    @property
    def logML_deriv_wderiv(self):
        self.calculate_logML_deriv_wderiv()
        return self._logML_deriv_wderiv
        

    @run_once
    def calculate_logML_deriv_wderiv(self):
        #Ljk0 = np.sqrt(0.5 / np.pi) * np.exp(self.logLjk())
        #Ljk1 = - np.sqrt(0.5 / np.pi) * np.exp(self.logLjk(derive = 1)) * self._y.reshape(self._n, 1)
        #mL   = self.ML.reshape(self._n, 1)
        #mL1  = self.ML_deriv.reshape(self._n, 1)
        #self._logML_deriv_wderiv = (Ljk1 / mL) - (Ljk0 * mL1 / np.square(mL))
        Ljk1_over_Ljk0 = np.exp(self.logLjk(derive = 1) - self.logLjk())
        self._logML_deriv_wderiv = - self.logML_wderiv \
            * (self._y.reshape(-1, 1) * Ljk1_over_Ljk0  + self.ML_deriv_over_ML.reshape(-1,1))
        return


    @property
    def logML_s2deriv(self):
        self.calculate_logML_s2deriv()
        return self._logML_s2deriv


    @run_once
    def calculate_logML_s2deriv(self):
        self._logML_s2deriv = self.ML_s2deriv_over_ML
        return


    @property
    def logML_deriv_s2deriv(self):
        self.calculate_logML_deriv_s2deriv()
        return self._logML_deriv_s2deriv


    @run_once
    def calculate_logML_deriv_s2deriv(self):
        self._logML_deriv_s2deriv = self.ML_deriv_s2deriv_over_ML \
            - (self.ML_deriv_over_ML * self.ML_s2deriv_over_ML)
        return
