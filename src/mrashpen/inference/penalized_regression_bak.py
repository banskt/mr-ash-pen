"""
Penalized regression method
wrapper for different minimization methods
"""

import numpy as np
from scipy import optimize as sp_optimize
import logging
import numbers

from ..models.normal_means_ash import NormalMeansASH
from ..models.plr_ash import PenalizedMrASH as PenMrASH
from ..utils.logs import MyLogger
from . import coordinate_descent_step as cd_step
from . import elbo as elbo_py

class PenalizedRegression:

    def __init__(self, method = 'L-BFGS-B', maxiter = 1000, 
                 display_progress = True, tol = 1e-9, options = None,
                 use_intercept = True,
                 is_prior_scaled = False,
                 wtol = 1e-2,
                 witer = 100,
                 optimize_w = True, optimize_s = True, 
                 calculate_elbo = False,
                 call_from_em = False, # just a hack to prevent printing termination info
                 debug = True):
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
        self._elbo_path = list()
        self._callback_count = 0
        self._obj_call_count = 0
        self._current_obj = 0
        self._final_obj = 0
        self._b = None
        self._binit = None
        self._wk = None
        self._wkinit = None
        self._s2 = None
        self._s2init = None
        self._theta = None
        self._intercept = None
        self._fitobj = None
        self._niter = 0
        self._optimize_w = optimize_w
        self._optimize_s = optimize_s
        self._wtol = wtol
        self._witer = witer
        self._debug = debug
        self._is_prior_scaled = is_prior_scaled
        self._use_intercept = use_intercept
        if debug:
            self.logger = MyLogger(__name__)
        else:
            self.logger = MyLogger(__name__, level = logging.INFO)
        self._calculate_elbo = calculate_elbo
        self._v2inv = 0
        self._call_from_em = call_from_em


    @property
    def theta(self):
        return self._theta


    @property
    def coef(self):
        return self._b


    @property
    def prior(self):
        return self._wk

    @property
    def residual_var(self):
        return self._s2


    @property
    def intercept(self):
        return self._intercept


    @property
    def success(self):
        return self._fitobj.success


    @property
    def obj_path(self):
        return self._hpath


    @property
    def elbo_path(self):
        return self._elbo_path


    @property
    def s2_path(self):
        return self._s2path


    @property
    def niter(self):
        if self._method == 'L-BFGS-B':
            return self._fitobj.nit


    @property
    def fun(self):
        return self._fitobj.fun


    @property
    def fitobj(self):
        return self._fitobj


    def objective(self, params):
        '''
        which parameters are being optimized
        '''
        b, wk, s2 = self.split_optparams(params)
        if not self._optimize_w: wk = self._wkinit
        if not self._optimize_s: s2 = self._s2init
        '''
        Get the objective function and gradients
        '''
        pmash = PenMrASH(self._X, self._y, b, np.sqrt(s2), wk, self._sk, dj = self._dj, 
                         debug = self._debug, is_prior_scaled = self._is_prior_scaled)
        obj = pmash.objective
        bgrad, wgrad, s2grad = pmash.gradients
        '''
        Combine gradients of all parameters for optimization
        Maximum p + k + 1 parameters: b, wk, s2
        '''
        lagrng = 1
        if self._optimize_w: obj += lagrng * np.sum(wk)
        grad = self.combine_optparams(bgrad, wgrad + lagrng, s2grad)
        '''
        Book-keeping
        '''
        self._obj_call_count += 1
        self._current_obj = obj
        self._current_s2  = s2
        return obj, grad


    def fit(self, X, y, sk, binit = None, winit = None, s2init = None):
        '''
        Dimensions of the problem
        '''
        n, p = X.shape
        k = sk.shape[0]
        ''' 
        This values will not change during the optimization
        '''
        self._X  = X
        self._sk = sk
        self._dj = np.sum(np.square(X), axis = 0)
        self._intercept = np.mean(y) if self._use_intercept else 0
        self._y = y - self._intercept
        self._v2inv = np.zeros((p, k))
        self._v2inv[:, 1:] = 1 / (self._dj.reshape(p, 1) + 1 / np.square(self._sk[1:]).reshape(1, k - 1))
        '''
        Initialization
        '''
        if binit is None:
            binit = np.zeros(p)
        if winit is None:
            winit = self.initialize_mixcoef(k)
        if s2init is None:
            s2init = np.var(self._y)
        assert(np.abs(np.sum(winit) - 1) < 1e-5)
        self._binit = binit
        self._wkinit = winit
        self._s2init = s2init
        '''
        Combine all parameters
        '''
        params = self.combine_optparams(binit, winit, s2init)
        '''
        Bounds for optimization
        '''
        bbounds = [(None, None) for x in binit]
        wbounds = [(1e-8, None) for x in winit]
        s2bound = [(1e-8, None)]
        # bounds can be used with L-BFGS-B.
        bounds = None
        if self._method == 'L-BFGS-B':
            bounds  = self.combine_optparams(bbounds, wbounds, s2bound)
        '''
        We need to pass s2init and winit as separate arguments
        in case they are not being optimized.
        _hpath: keeps track of the objective function.
        '''
        self._hpath  = list()
        self._s2path = list()
        self._elbo_path = list()
        self._callback_count = 0
        self._obj_call_count = 0
        plr_min = sp_optimize.minimize(self.objective, 
                                       params,
                                       #args = args, 
                                       method = self._method, 
                                       jac = True,
                                       bounds = bounds, 
                                       callback = self.callback,
                                       options = self._opts
                                       )
        '''
        Return values
        '''
        self._fitobj = plr_min
        bopt, wopt, s2opt = self.split_optparams(plr_min.x.copy())
        if self._optimize_w:
            wopt /= np.sum(wopt)
        else:
            wopt = winit
        if not self._optimize_s: s2opt = s2init
        self._theta = bopt
        pmash = PenMrASH(self._X, self._y, bopt, np.sqrt(s2opt), wopt, self._sk, dj = self._dj, 
                         debug = self._debug, is_prior_scaled = self._is_prior_scaled)
        self._b  = pmash.shrink_b
        self._wk = wopt
        self._s2 = s2opt
        '''
        Debug logging
        '''
        if not self._call_from_em:
            print (f"mr.ash.pen terminated at iteration {plr_min.nit}.")
        self.logger.debug(f'Number of iterations: {plr_min.nit}')
        self.logger.debug(f'Number of callbacks: {self._callback_count}')
        self.logger.debug(f'Number of function calls: {self._obj_call_count}')


    def split_optparams(self, optparams):
        n, p = self._X.shape
        k    = self._sk.shape[0]
        bj   = optparams[:p]. copy()
        if self._optimize_w:
            wk = optparams[p:p+k].copy()
        else:
            wk = None
        if self._optimize_s:
            s2idx = p + k if self._optimize_w else p
            s2 = optparams[s2idx].copy()
        else:
            s2 = None
        return bj, wk, s2


    def combine_optparams(self, bj, wk, s2):
        optparams = bj.copy()
        for val, is_included in zip([wk, s2], [self._optimize_w, self._optimize_s]): 
            if is_included:
                if isinstance(val, np.ndarray):
                    optparams = np.concatenate((optparams, val))
                elif isinstance(val, numbers.Real):
                    optparams = np.concatenate((optparams, np.array([val])))
                elif isinstance (val, list):
                    optparams += val
        return optparams


    def ebfit(self, X, y, sk, binit = None, winit = None, s2init = 1):
        # Do not optimize w and s2 with penalized regression
        self._optimize_w = False
        self._optimize_s = False
        outer_it = 0
        bbar = binit
        wbar = winit
        while wtol > self._wtol or it < self._witer:
            # Fit penalized regression for getting bbar
            self.fit(X, y, sk, binit = bbar, winit = wbar)
            bbar = self._b
            # Update wbar
            NMash = NormalMeansASH(bbar, np.sqrt(s2init), wbar, sk, debug = self._debug)
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
        #if self._callback_count == 80:
        #    print (f"Callback 80")
        self._hpath.append(self._current_obj)
        self._s2path.append(self._current_s2)
        if self._calculate_elbo:
            bopt, wopt, s2opt = self.split_optparams(params)
            if not self._optimize_w: wopt = self._wkinit
            if not self._optimize_s: s2opt = self._s2init
            wopt /= np.sum(wopt)
            pmash = PenMrASH(self._X, self._y, bopt, np.sqrt(s2opt), wopt, self._sk, dj = self._dj,
                             debug = self._debug, is_prior_scaled = self._is_prior_scaled)
            b  = pmash.shrink_b
            #elbo = cd_step.elbo(self._X, self._y, self._sk, b, wopt, s2opt, 
            #                        dj = self._dj, s2inv = self._v2inv)
            elbo = elbo_py.scalemix(self._X, self._y, self._sk, b, wopt, s2opt, 
                                    dj = self._dj, phijk = None, mujk = None, varjk = None, eps = 1e-8)
            self._elbo_path.append(elbo)
        self.logger.debug(f'Callback iteration {self._callback_count}')


    def initialize_mixcoef(self, k):
        w = np.zeros(k)
        w[1:(k-1)] = np.repeat(1/(k-1), (k - 2))
        w[k-1] = 1 - np.sum(w)
        return w
