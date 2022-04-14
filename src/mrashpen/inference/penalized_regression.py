"""
Penalized regression method
wrapper for different minimization methods
"""

import numpy as np
from scipy import optimize as sp_optimize
from scipy.optimize import OptimizeResult as spOptimizeResult
import logging
import numbers

from ..models.normal_means_ash import NormalMeansASH
from ..models.plr_ash          import PenalizedMrASH as PenMrASH
from ..models                  import mixture_gaussian as mix_gauss
from ..utils.logs              import MyLogger
from . import coordinate_descent_step as cd_step
from . import elbo as elbo_py

from libmrashpen_plr_mrash import plr_mrash as flib_penmrash
from libmrashpen_lbfgs_driver import lbfgsb_driver as flib_lbfgsb_driver


def softmax(x, base = np.exp(1)):
    if base is not None:
        beta = np.log(base)
        x = x * beta
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x, axis = 0, keepdims = True)


class PenalizedRegression:

    def __init__(self, method = 'L-BFGS-B', maxiter = 2000,
                 display_progress = True, tol = 1e-9, options = None,
                 use_intercept = True,
                 is_prior_scaled = True,
                 optimize_b = True, optimize_w = True, optimize_s = True, 
                 calculate_elbo = False,
                 call_from_em = False, # just a hack to prevent printing termination info
                 prior_optim_method = 'softmax', # can be softmax or mixsqp
                 unshrink_method = 'newton', # can be 'newton' (newton-raphson-inversion) or 'heuristic'
                 function_call = 'fortran',
                 lbfgsb_call = 'fortran',
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
        self._optimize_b = optimize_b
        self._optimize_w = optimize_w
        self._optimize_s = optimize_s
        self._debug = debug
        self._is_prior_scaled = is_prior_scaled
        self._use_intercept = use_intercept
        if debug:
            self.logger = MyLogger(__name__)
        else:
            self.logger = MyLogger(__name__, level = logging.INFO)
        self._calculate_elbo = calculate_elbo
        self._call_from_em = call_from_em
        self._w_use_mixsqp       = True if prior_optim_method == 'mixsqp' else False
        self._w_use_softmax      = True if prior_optim_method == 'softmax' else False
        self._b_use_Minverse     = True if unshrink_method == 'newton' else False
        self._b_use_heuristic    = True if unshrink_method == 'heuristic' else False
        self._f_use_fortran      = True if function_call == 'fortran' else False
        self._lbfgsb_use_fortran = True if lbfgsb_call == 'fortran' else False
        return


    @property
    def theta(self):
        return self._theta


    @property
    def coef(self):
        return self._coef


    @property
    def prior(self):
        return self._prior


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
    def prior_path(self):
        return self._prior_path


    @property
    def theta_path(self):
        return self._theta_path


    @property
    def coef_path(self):
        return self._coef_path


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


    def pmash_obj_gradients(self, b, sigma, wk):
        if self._f_use_fortran:
            '''
            Call function objective and gradient from FORTRAN
            '''
            djinv = 1 / self._dj
            s2 = sigma * sigma
            obj, bgrad, wgrad, s2grad \
                = flib_penmrash.plr_obj_grad_shrinkop(self._X, self._y, b, s2, wk, self._sk, djinv)
        else:
            '''
            Call function objective and gradient from Python
            '''
            pmash = PenMrASH(self._X, self._y, b, sigma, wk, self._sk, dj = self._dj,
                             debug = self._debug, is_prior_scaled = self._is_prior_scaled)
            obj = pmash.objective
            bgrad, wgrad, s2grad = pmash.gradients
        return obj, bgrad, wgrad, s2grad


    def objective(self, params):
        '''
        which parameters are being optimized
        b  -> theta 
        ak -> parameters controlling mixture coefficients wk
              w = ak / sum(ak)  | for mixsqp
              w = softmax(ak)   | for softmax
        s2 -> residual variance
        '''
        b, ak, s2 = self.split_optparams(params)
        # do not scale ak to prior if using mixsqp
        wk = ak.copy()
        if self._w_use_softmax:
            wk = softmax(ak, base = self._softmax_base)
        '''
        Get the objective function and gradients
        '''
        obj, bgrad, wgrad, s2grad = self.pmash_obj_gradients(b, np.sqrt(s2), wk)
        '''
        Combine gradients of all parameters for optimization
        Maximum p + k + 1 parameters: b, wk, s2
        '''
        if self._optimize_w:
            if self._w_use_mixsqp:
                lagrng = 1
                obj   += lagrng * np.sum(wk)
                wgrad += lagrng
            elif self._w_use_softmax:
                k      = wk.shape[0]
                akjac  = np.log(self._softmax_base) * wk.reshape(-1, 1) * (np.eye(k) - wk)
                wgrad  = np.sum(wgrad * akjac, axis = 1)
        grad = self.combine_optparams(bgrad, wgrad, s2grad)
        '''
        Book-keeping
        '''
        self._obj_call_count += 1
        self._current_obj   = obj
        self._current_s2    = s2
        self._current_ak    = ak
        self._current_theta = b
        return obj, grad


    def initialize_params(self, binit, winit, s2init, t0init, is_coef):
        n, p = self._X.shape
        k    = self._sk.shape[0]
        '''
        if binit is not given, use blind initialization (all zero)
        if binit is given, could be either coef or theta
        '''
        if binit is None:
            theta_init = np.zeros(p)
            if s2init is None: s2init = np.var(self._y)
            if winit  is None: winit  = self.initialize_mixcoef(k)
            should_unshrink = False
        else:
            if is_coef:
                ### binit are coefficients, we have to find theta = M^{-1}(binit)
                if s2init is None: s2init = np.var(self._y - np.dot(self._X, binit))
                if winit  is None: winit  = mix_gauss.emfit(binit, self._sk)
                if t0init is None: t0init = np.zeros(p)
                if self._b_use_Minverse:
                    pmash = PenMrASH(self._X, self._y, t0init, np.sqrt(s2init), winit, self._sk, dj = self._dj,
                                     debug = self._debug, is_prior_scaled = self._is_prior_scaled)
                    theta_init = pmash.unshrink_b(binit)
                elif self._b_use_heuristic:
                    theta_init = self.coef_to_theta(binit)
            else:
                ### binit = theta, therefore coef = M(binit)
                ### if theta_init is given, we also expect winit and s2init to be given
                ### TO-DO: find better winit and s2init if they are not given
                theta_init = binit.copy()
                if winit  is None: winit  = self.initialize_mixcoef(k)
                if s2init is None: s2init = np.var(self._y)
        '''
        Finally, obtain the ak from mixture coefficients
        '''
        if self._w_use_mixsqp:
            akinit = winit.copy()
        elif self._w_use_softmax:
            akinit = np.log(winit + 1e-8) / np.log(self._softmax_base)
            winit  = softmax(akinit, base = self._softmax_base)
        return theta_init, winit, akinit, s2init


    def fit(self, X, y, sk, binit = None, winit = None, s2init = None,
            inv_binit     = None,    # could be used for initializing theta for Minverse
            is_binit_coef = True,    # set this False when initializing with theta
            softmax_base = np.exp(1) # set this to large values for faster shrinkage of sparsity
            ):
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
        self._intercept = np.mean(y) if self._use_intercept else 0
        self._y = y - self._intercept
        self._softmax_base = softmax_base
        self._dj = np.sum(np.square(self._X), axis = 0)
        '''
        Initialization
        '''
        binit, wkinit, akinit, s2init = self.initialize_params(binit, winit, s2init, inv_binit, is_binit_coef)
        self._binit  = binit
        self._akinit = akinit
        self._s2init = s2init
        self._wkinit = wkinit
        '''
        Minimization
        '''
        if self._method == 'L-BFGS-B' and self._lbfgsb_use_fortran:
            '''
            Use Fortran minimization
            '''
            self.fit_fortran()
        else:
            '''
            Use Scipy minimization
            '''
            self.fit_python()
        return


    def fit_fortran(self):
        '''
        set options
        '''
        smlb    = np.log(self._softmax_base)
        iprint  = 1 if self._opts['disp'] else -1
        factr   = self._opts['ftol'] / np.finfo(float).eps
        pgtol   = self._opts['gtol']
        maxiter = self._opts['maxiter']
        maxfun  = self._opts['maxfun']
        '''
        number of parameters to be optimized
        (required as input to Fortran)
        '''
        nparams = 0
        if self._optimize_b: nparams += self._X.shape[1]
        if self._optimize_w: nparams += self._sk.shape[0]
        if self._optimize_s: nparams += 1
        '''
        Call L-BFGS-B drive
        '''
        bopt, wkopt, s2opt, obj, grad, nfev, niter, task = \
            flib_lbfgsb_driver.min_plr_shrinkop(self._X, self._y, 
                                                self._binit, self._wkinit, self._s2init,
                                                self._sk, nparams, 
                                                self._optimize_b, self._optimize_w, self._optimize_s,
                                                smlb, 10, iprint, factr, pgtol, 
                                                maxiter, maxfun)
        '''
        Set output values
        '''
        xopt = self.combine_optparams(bopt, wkopt, s2opt)
        coef = self.theta_to_coef(bopt, wkopt, s2opt)
        self._theta = bopt
        self._coef  = coef
        self._prior = wkopt
        self._s2    = s2opt
        '''
        Status from task message
        '''
        task_str = task.strip(b'\x00').strip()
        if task_str.startswith(b'CONV'):
            warnflag = 0
        elif nfev > maxfun or niter >= maxiter:
            warnflag = 1
        else:
            warnflag = 2
        '''
        TO-DO: These values are not returned yet
        '''
        self._hpath      = list()
        self._s2path     = list()
        self._prior_path = list()
        self._coef_path  = list()
        self._theta_path = list()
        self._elbo_path  = list()
        self._fitobj     = spOptimizeResult(fun = obj, jac = grad, nfev = nfev,
                                            njev = nfev, nit = niter, status = warnflag,
                                            message = task_str.decode(), x = xopt, 
                                            success = (warnflag == 0),
                                            hess_inv = None)

        '''
        Debug logging
        '''
        if not self._call_from_em:
            print (f"mr.ash.pen terminated at iteration {niter}.")
            print (task_str.decode())
        self.logger.debug(f'Number of iterations: {niter}')
        self.logger.debug(f'Number of function calls: {nfev}')
        self.logger.debug(f'Message from Fortran routine:\n{task}')
        return
            


    def fit_python(self):
        ''' 
        This values will not change during the optimization
        '''
        n, p = self._X.shape
        k = self._sk.shape[0]
        self._v2inv = np.zeros((p, k))
        self._v2inv[:, 1:] = 1 / (self._dj.reshape(p, 1) + 1 / np.square(self._sk[1:]).reshape(1, k - 1))
        '''
        Combine all parameters
        '''
        params = self.combine_optparams(self._binit, self._akinit, self._s2init)
        '''
        Bounds for optimization
        '''
        bbounds = [(None, None) for x in self._binit]
        abounds = [(1e-8, None) for x in self._akinit]
        if self._w_use_softmax: abounds = [(None, None) for x in self._akinit]
        s2bound = [(1e-8, None)]
        # bounds can be used with L-BFGS-B.
        bounds = None
        if self._method == 'L-BFGS-B':
            bounds  = self.combine_optparams(bbounds, abounds, s2bound)
        '''
        We need to pass s2init and akinit as separate arguments
        in case they are not being optimized.
        hpath: keeps track of the objective function.
        '''
        self._hpath      = list()
        self._s2path     = list()
        self._prior_path = list()
        self._coef_path  = list()
        self._theta_path = list()
        self._elbo_path  = list()
        self._callback_count = 0
        self._obj_call_count = 0
        plr_min = sp_optimize.minimize(self.objective, 
                                       params,
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
        bopt, akopt, s2opt = self.split_optparams(plr_min.x.copy())
        prior = self.wparam_to_mixcoef(akopt)
        coef  = self.theta_to_coef(bopt, prior, s2opt)
        self._theta = bopt
        self._coef  = coef
        self._prior = prior
        self._s2    = s2opt
        '''
        Debug logging
        '''
        if not self._call_from_em:
            print (f"mr.ash.pen terminated at iteration {plr_min.nit}.")
        self.logger.debug(f'Number of iterations: {plr_min.nit}')
        self.logger.debug(f'Number of callbacks: {self._callback_count}')
        self.logger.debug(f'Number of function calls: {self._obj_call_count}')
        return


    def split_optparams(self, optparams):
        n, p = self._X.shape
        k    = self._sk.shape[0]
        idx  = 0
        if self._optimize_b:
            bj = optparams[:p]. copy()
            idx += p
        else:
            bj = self._binit
        if self._optimize_w:
            ak = optparams[idx:idx+k].copy()
            idx += k
        else:
            ak = self._akinit
        if self._optimize_s:
            s2 = optparams[idx].copy()
        else:
            s2 = self._s2init
        return bj, ak, s2


    def combine_optparams(self, bj, ak, s2):
        optparams = np.array([])
        if any([isinstance(x, list) for x in [bj, ak, s2]]):
            optparams = list()
        for val, is_included in zip([bj, ak, s2], [self._optimize_b, self._optimize_w, self._optimize_s]): 
            if is_included:
                if isinstance(val, np.ndarray):
                    optparams = np.concatenate((optparams, val))
                elif isinstance(val, numbers.Real):
                    optparams = np.concatenate((optparams, np.array([val])))
                elif isinstance (val, list):
                    optparams += val
        return optparams


    def callback(self, params):
        self._callback_count += 1
        # The objective function can be called multiple times in between callbacks
        # We append to paths only during callbacks
        self._hpath.append(self._current_obj)
        self._s2path.append(self._current_s2)
        self._prior_path.append(self._current_ak)
        self._theta_path.append(self._current_theta)
        if self._calculate_elbo:
            bopt, akopt, s2opt = self.split_optparams(params)
            wk    = self.wparam_to_mixcoef(akopt)
            coef  = self.theta_to_coef(bopt, wk, s2opt)
            elbo  = cd_step.elbo(self._X, self._y, self._sk, coef, wk, s2opt, 
                                    dj = self._dj, s2inv = self._v2inv)
            #elbo = elbo_py.scalemix(self._X, self._y, self._sk, coef, wopt, s2opt, 
            #                        dj = self._dj, phijk = None, mujk = None, varjk = None, eps = 1e-8)
            self._coef_path.append(coef)
            self._elbo_path.append(elbo)
        self.logger.debug(f'Callback iteration {self._callback_count}')


    def initialize_mixcoef(self, k, sparsity = 0.8):
        w = np.zeros(k)
        w[0] = sparsity
        w[1:(k-1)] = np.repeat((1 - w[0])/(k-1), (k - 2))
        w[k-1] = 1 - np.sum(w)
        return w


    def wparam_to_mixcoef(self, ak):
        wk = ak.copy()
        if self._w_use_mixsqp:
            wk = ak / np.sum(ak)
        elif self._w_use_softmax:
            wk = softmax(ak, base = self._softmax_base)
        return wk


    def theta_to_coef(self, bopt, wk, s2):
        pmash = PenMrASH(self._X, self._y, bopt, np.sqrt(s2), wk, self._sk, dj = self._dj,
                         debug = self._debug, is_prior_scaled = self._is_prior_scaled)
        coef  = pmash.shrink_b
        return coef


    def coef_to_theta(self, b):
        n, p = self._X.shape
        r    = self._y - np.mean(self._y) - np.dot(self._X, b)
        rj   = r.reshape(n, 1) + self._X * b.reshape(1, p)
        theta = np.einsum('ij,ij->j', self._X, rj) / self._dj
        return theta
