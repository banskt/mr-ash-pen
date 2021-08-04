"""
Penalized regression method
wrapper for different minimization methods
"""

import numpy as np
from scipy import optimize as sp_optimize
import logging

from ..models.normal_means_ash import NormalMeansASH
from ..utils.logs import MyLogger

class PenalizedRegression:

    def __init__(self, method = 'L-BFGS-B', maxiter = 1000, 
                 display_progress = True, tol = 1e-9, options = None, 
                 wtol = 1e-2, witer = 100,
                 optimize_w = False, optimize_s = False, debug = True):
        self._method = method
        self._opts   = options
        if options is None:
            if method == 'L-BFGS-B':
                self._opts = {'maxiter': maxiter, # Maximum number of iterations
                              'maxfun': maxiter * 10, # Maximum number of function evaluations
                              'ftol': tol, # Function tolerance. stop when ``(f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= ftol``.
                              'gtol': tol, # Gradient tolerance. stop when ``max{|proj g_i | i = 1, ..., n} <= gtol``
                              'disp': display_progress
                              }
            elif method == 'CG':
                self._opts = {'maxiter': maxiter, # Maximum number of iterations
                              'disp': display_progress,
                              'gtol': tol, # Gradient norm must be less than gtol before successful termination.
                             }
            else:
                self._opts = {'maxiter': maxiter, # Maximum number of iterations
                              'disp': display_progress
                              }
        self._hpath  = list()
        self._callback_count = 0
        self._obj_call_count = 0
        self._current_obj = 0
        self._b = None
        self._wk = None
        self._success = False
        self._niter = 0
        self._final_obj = 0
        self._optimize_w = optimize_w
        self._optimize_s = optimize_s
        self._wtol = wtol
        self._witer = witer
        if debug:
            self.logger = MyLogger(__name__)
        else:
            self.logger = MyLogger(__name__, level = logging.INFO)


    @property
    def coef(self):
        return self._b


    @property
    def wk(self):
        return self._wk


    @property
    def success(self):
        return self._fitobj.success


    @property
    def obj_path(self):
        return self._hpath


    @property
    def niter(self):
        return self._fitobj.nit


    @property
    def fun(self):
        return self._fitobj.fun


    @property
    def fitobj(self):
        return self._fitobj



    def shrinkage_operator(self, NMash):
        '''
        posterior expectation of b under NM model
        calculated using Tweedie's formula
        '''
        n       = NMash.y.shape[0]
        S       = NMash.y + NMash.yvar * NMash.logML_deriv
        S_bgrad = 1       + NMash.yvar * NMash.logML_deriv2
        S_wgrad = NMash.yvar.reshape(n, 1) * NMash.logML_deriv_wderiv
        return S, S_bgrad, S_wgrad


    def hS_operator(self, NMash):
        n       = NMash.y.shape[0]
        s2      = NMash.yvar
        h       = - NMash.logML  - 0.5 * s2 * NMash.logML_deriv * NMash.logML_deriv
        # Gradient with respect to b
        h_bgrad = - NMash.logML_deriv  - s2 * NMash.logML_deriv * NMash.logML_deriv2
        # Gradient with repect to w
        h_wgrad = - NMash.logML_wderiv - s2.reshape(n, 1) * NMash.logML_deriv.reshape(n, 1) * NMash.logML_deriv_wderiv
        h_wgrad = np.sum(h_wgrad, axis = 0)
        return h, h_bgrad, h_wgrad


    def objective(self, params, X, y, s2init, winit, sk):
        '''
        The objective function is written as
        H(Sb, f, s^2) = (0.5 / s^2) || y - X.Sb||^2 } + h(b, w)
        where, h(b, w) := sum_j (d_j / s^2) rho[g, s/sqrt(dj)] (Sb)
        '''
        dj = np.sum(np.square(X), axis = 0)
        n, p = X.shape
        k = sk.shape[0]

        '''
        which parameters are being optimized
        '''
        b = params[:p]
        wk = params[p:p+k] if self._optimize_w else winit
        s2idx = p+k  if self._optimize_w else p
        s2 = params[s2idx] if self._optimize_s else s2init
        sj = np.sqrt(s2 / dj)


        NMash = NormalMeansASH(b, sj, wk, sk)

        '''
        S(b) and h(S(b))
        '''
        Sb, Sb_bgrad, Sb_wgrad = self.shrinkage_operator(NMash)
        hS, hS_bgrad, hS_wgrad = self.hS_operator(NMash)

        '''
        Objective function
        '''
        r = y - np.dot(X, Sb)
        if self._optimize_w:
            lagrng = 1

        obj = (0.5 * np.sum(np.square(r)) / s2) + np.sum(hS)
        #self.logger.debug(f'Objective without Lagrangian: {obj}')
        # Add Lagrange multiplier if optimizing for wk
        if self._optimize_w:
            obj += lagrng * np.sum(wk)
            #self.logger.debug(f'Objective with Lagrangian: {obj}')
            #self.logger.debug(f'Sum of wk: {np.sum(wk)}')
        
        '''
        Gradients
        '''
        grad = - (np.dot(r.T, X) * Sb_bgrad / s2) + np.sum(hS_bgrad)
        if self._optimize_w:
            wgrad  = - np.dot(np.dot(r.T, X), Sb_wgrad) / s2 + hS_wgrad + lagrng
            grad = np.concatenate((grad, wgrad))

        self._obj_call_count += 1
        self._current_obj = obj
        #self.logger.debug(f'Objective evaluation count {self._obj_call_count}. Objective: {obj}')

        return obj, grad
        

    def fit(self, X, y, sk, binit = None, winit = None, s2init = 1):
        n, p = X.shape
        k = sk.shape[0]
        if binit is None:
            binit = np.zeros(p)
        if winit is None:
            winit = self.initialize_mixcoef(k)
        assert(np.abs(np.sum(winit) - 1) < 1e-5)

        params = binit.copy()
        bounds = [(None, None) for x in params]
        if self._optimize_w: 
            params = np.concatenate((params, winit))
            bounds += [(0, None) for x in winit]
        if self._optimize_s: 
            params = np.concatenate((params, s2init))
            bounds.append((0,1))

        # cannot use bounds with CG.
        if self._method == 'CG': bounds = None

        args = X, y, s2init, winit, sk
        self._hpath = list()
        self._callback_count = 0
        self._obj_call_count = 0
        plr_min = sp_optimize.minimize(self.objective, 
                                       params,
                                       args = args, 
                                       method = self._method, 
                                       jac=True,
                                       bounds = bounds, 
                                       callback = self.callback,
                                       options = self._opts
                                       )

        self._fitobj = plr_min
        bopt = plr_min.x[:p].copy()
        wopt = plr_min.x[p:p+k].copy() if self._optimize_w else winit
        wopt /= np.sum(wopt)
        NMash = NormalMeansASH(bopt, np.sqrt(s2init), wopt, sk)
        self._b = self.shrinkage_operator(NMash)[0]
        self._wk = wopt
        self._hpath.append(self._current_obj)

        self.logger.debug(f'Number of iterations: {plr_min.nit}')
        self.logger.debug(f'Number of callbacks: {self._callback_count}')
        self.logger.debug(f'Number of function calls: {self._obj_call_count}')



    def ebfit(self, X, y, sk, binit = None, winit = None, s2init = 1):
        # Do not optimize w with penalized regression
        self._optimize_w = False
        wtol = 1e8
        it = 0
        bbar = binit
        wbar = winit
        while wtol > self._wtol or it < self._witer:
            # Fit penalized regression for getting bbar
            self.fit(X, y, sk, binit = bbar, winit = wbar)
            bbar = self._b
            # Update wbar
            NMash = NormalMeansASH(bbar, np.sqrt(s2init), wbar, sk)
            phijk, mujk, varjk = NMash.posterior()
            wbar_new = np.sum(phijk, axis = 0) / phijk.shape[0]
            # Tolerance
            wtol = np.min(np.abs(wbar_new - wbar))
            wbar = wbar_new.copy()
            it += 1

        # Set values after convergence
        self._wk = wbar



    def callback(self, params):
        self._callback_count += 1
        self._hpath.append(self._current_obj)
        #self.logger.debug(f'Callback iteration {self._callback_count}')


    def initialize_mixcoef(self, k):
        w = np.zeros(k)
        w[1:(k-1)] = np.repeat(1/(k-1), (k - 2))
        w[k-1] = 1 - np.sum(w)
        return w

