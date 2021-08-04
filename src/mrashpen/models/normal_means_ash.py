"""
Class for normal means model
with ASH prior
    p(y | b, s^2) = N(y | b, s^2)
    p(b) = sum_k w_k N(b | 0, s_k^2)
"""

import numpy as np

class NormalMeansASH:

    def __init__(self, y, s, wk, sk):
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
        if not isinstance(s, np.ndarray):
            self._s = np.repeat(s, self._n)
        self._s2 = np.square(self._s)
        

    @property
    def y(self):
        return self._y


    @property
    def yvar(self):
        return self._s2

    @property
    def ML(self):
        '''
        marginal likelihood under mixture prior
        p(y | f, s^2) = sum_k wk * N(y | 0, s^2 + s_k^2)
        '''
        L = np.exp(self.logLjk()) # N x K
        marglik = np.sqrt(0.5 / np.pi) * np.dot(L, self._wk)
        return marglik


    @property
    def ML_deriv_over_y(self):
        '''
        returns ML_deriv(y) / y 
        ok even if y = 0
        '''
        L = np.exp(self.logLjk(derive = 1)) # N x K
        deriv_over_y = - np.sqrt(0.5 / np.pi) * np.dot(L, self._wk)
        return deriv_over_y


    @property
    def ML_deriv(self):
        return self.ML_deriv_over_y * self._y


    @property
    def ML_deriv2(self):
        L = np.exp(self.logLjk(derive = 2)) # N x K
        deriv2 = np.sqrt(0.5 / np.pi) * np.dot(L, self._wk) * np.square(self._y) + self.ML_deriv_over_y
        return deriv2


    def logLjk(self, derive = 0):
        '''
        this is one part of the posterior in normal means model. LogLjk is defined as:    
            p(y | f, s2)   =   (1 / sqrt(2 * pi)) * sum_k [w_k * exp(logLjk)]            # derive = 0
            p'(y | f, s2)  = - (y / sqrt(2 * pi)) * sum_k [w_k * exp(logLjk)]            # derive = 1 (first derivative)
            p''(y | f, s2) = (y^2 / sqrt(2 * pi)) * sum_k [w_k * exp(logLjk)] + p' / y   # derive = 2 
        returns N x K matrix
        '''
        s2  = np.square(self._s).reshape(self._n, 1)
        sk2 = np.square(self._sk).reshape(1, self._k)
        y2  = np.square(self._y).reshape(self._n, 1)
        # N x K length vector of posterior variances
        v2 = s2 + sk2
        if derive == 0:
            logL = -0.5 * (np.log(v2) + y2 / v2)       # N x K matrix
        elif derive == 1:
            logL = -0.5 * (3 * np.log(v2) + (y2 / v2)) # N x K matrix
        elif derive == 2:
            logL = -0.5 * (5 * np.log(v2) + (y2 / v2)) # N x K matrix
        return logL


    def posterior(self):
        s2  = np.square(self._s).reshape(self._n, 1)
        sk2 = np.square(self._sk).reshape(1, self._k)
        y2  = np.square(self._y).reshape(self._n, 1)
        v2jk  = s2 + sk2
        mujk  = self._y.reshape(self._n, 1) * sk2 / v2jk
        varjk = s2 * sk2 / v2jk

        logLjk = -0.5 * (np.log(v2jk) + y2 / v2jk)
        phijk  = np.sqrt(0.5 / np.pi) * self._wk * np.exp(logLjk)
        phijk /= np.sum(phijk, axis = 1).reshape(self._n, 1)
        return phijk, mujk, varjk


    @property
    def logML(self):
        return np.log(self.ML)


    @property
    def logML_deriv(self):
        return self.ML_deriv / self.ML


    @property
    def logML_deriv2(self):
        fy    = self.ML
        fy_d1 = self.ML_deriv
        fy_d2 = self.ML_deriv2
        l2 = ((fy * fy_d2) - (fy_d1 * fy_d1)) / (fy * fy)
        return l2


    @property
    def logML_wderiv(self):
        return np.sqrt(0.5 / np.pi) * np.exp(self.logLjk()) / self.ML.reshape(self._n, 1)


    @property
    def logML_deriv_wderiv(self):
        Ljk0 = np.sqrt(0.5 / np.pi) * np.exp(self.logLjk())
        Ljk1 = - np.sqrt(0.5 / np.pi) * np.exp(self.logLjk(derive = 1)) * self._y.reshape(self._n, 1)
        mL   = self.ML.reshape(self._n, 1)
        mL1  = self.ML_deriv.reshape(self._n, 1)
        l2 = (Ljk1 / mL) - (Ljk0 * mL1 / np.square(mL))
        return l2
