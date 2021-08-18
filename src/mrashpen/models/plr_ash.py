'''
Objective function and its derivatives
for Mr.ASH
'''

import numpy as np
import logging

from ..models.normal_means_ash import NormalMeansASH
from ..utils.logs import MyLogger

class PenalizedMrASH:

    def __init__(self, X, y, b, sigma, wk, sk, dj = None, debug = True):
        self._X = X
        self._y = y
        self._b = b
        self._n, self._p = X.shape
        self._k = wk.shape[0]
        self._s = sigma
        self._s2 = np.square(self._s)
        self._wk = wk
        self._sk = sk
        self.set_Xvar(dj)
        if debug:
            self.logger = MyLogger(__name__)
        else:
            self.logger = MyLogger(__name__, level = logging.INFO)
        self._calculate_plr_once = False



    def run_once(f):
        def wrapper(*args, **kwargs):
            if not wrapper.has_run:
                wrapper.has_run = True
                return f(*args, **kwargs)
        wrapper.has_run = False
        return wrapper


    '''
    Allows setting dj externally to avoid calculating it multiple times
    '''
    def set_Xvar(self, dj = None):
        if dj is None:
            self._dj = np.sum(np.square(self._X), axis = 0)
        else:
            self._dj = dj
        self._vj = np.sqrt(self._s2 / self._dj)
        return


    def shrinkage_operator(self, nm):
        '''
        posterior expectation of b under NM model
        calculated using Tweedie's formula

        Returns shrinkage operator M(b)
        Dimensions:
            M: vector of size P
            M_bgrad: vector of size P
            M_wgrad: matrix of size P x K
            M_sgrad: vector of size P
        '''
        M       = nm.y + nm.yvar * nm.logML_deriv
        M_bgrad = 1       + nm.yvar * nm.logML_deriv2
        M_wgrad = nm.yvar.reshape(-1, 1) * nm.logML_deriv_wderiv
        M_sgrad = (nm.logML_deriv + nm.yvar * nm.logML_deriv_s2deriv) / self._dj
        return M, M_bgrad, M_wgrad, M_sgrad


    def penalty_operator(self, nm):
        '''
        Returns the penalty operator, defined as sum_j lambda_j = rho(M(b)) / vj^2
        Dimensions:
            lambdaj: vector of size P
            l_bgrad: vector of size P
            l_wgrad: vector of size K
            l_sgrad: vector of size P 
        Note: lambdaj to be summed outside this function for sum_j lambda_j
              l_sgrad to be summed outside this function for sum_j d/ds2 lambda_j
        '''
        lambdaj = - nm.logML - 0.5 * nm.yvar * np.square(nm.logML_deriv)
        lsum    = np.sum(lambdaj)
        # Gradient with respect to b
        l_bgrad = - nm.logML_deriv  - nm.yvar * nm.logML_deriv * nm.logML_deriv2
        # Gradient with repect to w
        v2_ld_ldwd = nm.yvar.reshape(-1, 1) * nm.logML_deriv.reshape(-1, 1) * nm.logML_deriv_wderiv
        l_wgrad = - nm.logML_wderiv - v2_ld_ldwd
        l_wgrad = np.sum(l_wgrad, axis = 0)
        # Gradient with respect to sigma2
        v2_ld_lds2d = nm.yvar * nm.logML_deriv * nm.logML_deriv_s2deriv
        l_sgrad = - (nm.logML_s2deriv - 0.5 * np.square(nm.logML_deriv) - v2_ld_lds2d) / self._dj
        return lambdaj, l_bgrad, l_wgrad, l_sgrad


    def calculate_plr_objective(self):
        if not self._calculate_plr_once:
            self.logger.debug("Calculating PLR objective")
            '''
            Initiate the Normal Means model
            which is used for the calculation
            '''
            nmash = NormalMeansASH(self._b, self._vj, self._wk, self._sk)
            '''
            M(b) and lambda_j
            '''
            Mb, Mb_bgrad, Mb_wgrad, Mb_s2grad = self.shrinkage_operator(nmash)
            lj, l_bgrad,  l_wgrad,  l_s2grad  = self.penalty_operator(nmash)
            '''
            Objective function
            '''
            r = self._y - np.dot(self._X, Mb)
            rTr  = np.sum(np.square(r))
            rTX  = np.dot(r.T, self._X)
            obj  = (0.5 * rTr / self._s2) + np.sum(lj)
            obj += 0.5 * (self._n - self._p) * np.log(2 * np.pi * self._s2)
            '''
            Gradients
            '''
            bgrad  = - (rTX * Mb_bgrad / self._s2) + l_bgrad
            wgrad  = - np.dot(rTX, Mb_wgrad) / self._s2  + l_wgrad
            s2grad = - 0.5 * rTr / (self._s2 * self._s2) \
                     - np.dot(rTX, Mb_s2grad) / self._s2 \
                     + np.sum(l_s2grad) \
                     + 0.5 * (self._n - self._p) / self._s2
    
            self._objective = obj
            self._bgrad = bgrad
            self._wgrad = wgrad
            self._s2grad = s2grad
        self._calculate_plr_once = True
        return


    def set_s2_eps(self, eps):
        self._s2 += eps
        self._s = np.sqrt(self._s2)
        self.set_Xvar(dj = self._dj)
        return


    @property
    def objective(self):
        self.calculate_plr_objective()
        return self._objective


    @property
    def gradients(self):
        self.calculate_plr_objective()
        return self._bgrad, self._wgrad, self._s2grad
