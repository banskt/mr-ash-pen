import numpy as np
import collections
from .penalized_regression import PenalizedRegression as PLR
from . import elbo as elbo_py
from . import coordinate_descent_step as cd_step
from ..models.normal_means_ash_scaled import NormalMeansASHScaled


RES_FIELDS = ['theta', 'coef', 'prior', 'residual_var', 'intercept',
              'elbo_path', 'obj_path', 'niter', 'plr1', 'plr2', 'plr3', 'plr4']
class ResInfo(collections.namedtuple('_ResInfo', RES_FIELDS)):
    __slots__ = ()


def method_fixseq_conv(X, y, sk, winit, binit = None, s2init = 1, 
        maxiter = 200,
        plr_maxiter = 4000,
        calculate_elbo = False,
        debug = False,
        display_progress = False,
        epstol = 1e-8):

    n, p = X.shape
    k    = sk.shape[0]
    intercept = np.mean(y)
    y    = y - intercept
    dj   = np.sum(np.square(X), axis = 0)

    outer_elbo_path = list()

    ## Update b and w
    plr1 = PLR(method = 'L-BFGS-B', optimize_b = True, optimize_w = True, optimize_s = False, is_prior_scaled = True,
              debug = debug, display_progress = display_progress, calculate_elbo = calculate_elbo, maxiter = plr_maxiter)
    plr1.fit(X, y, sk, binit = binit, winit = winit, s2init = s2init)

    ## Update s
    plr2 = PLR(method = 'L-BFGS-B', optimize_b = True, optimize_w = False, optimize_s = True, is_prior_scaled = True,
               debug = debug, display_progress = display_progress, calculate_elbo = calculate_elbo, maxiter = plr_maxiter)
    plr2.fit(X, y, sk, binit = plr1.theta, winit = plr1.prior, s2init = plr1.residual_var, unshrink_binit = False)

    ## Update w
    plr3 = PLR(method = 'L-BFGS-B', optimize_b = False, optimize_w = True, optimize_s = False, is_prior_scaled = True,
               debug = debug, display_progress = display_progress, calculate_elbo = calculate_elbo, maxiter = plr_maxiter)
    plr3.fit(X, y, sk, binit = plr2.theta, winit = plr2.prior, s2init = plr2.residual_var, unshrink_binit = False)

    ## Update b and w
    plr4 = PLR(method = 'L-BFGS-B', optimize_b = True, optimize_w = True, optimize_s = False, is_prior_scaled = True,
               debug = debug, display_progress = display_progress, calculate_elbo = calculate_elbo, maxiter = plr_maxiter)
    plr4.fit(X, y, sk, binit = plr3.theta, winit = plr3.prior, s2init = plr3.residual_var, unshrink_binit = False)

    obj_path  = plr1.obj_path + plr2.obj_path + plr3.obj_path + plr4.obj_path
    elbo_path = list()
    if calculate_elbo:
        elbo_path = plr1.elbo_path + plr2.elbo_path + plr3.elbo_path + plr4.elbo_path

    res = ResInfo(theta = plr4.theta,
                  coef = plr4.coef,
                  prior = plr4.prior,
                  residual_var = plr4.residual_var,
                  intercept = intercept,
                  elbo_path = elbo_path,
                  obj_path = obj_path,
                  niter = len(obj_path),
                  plr1 = plr1,
                  plr2 = plr2, 
                  plr3 = plr3,
                  plr4 = plr4)

    return res


def method_fixseq(X, y, sk, winit, binit = None, s2init = 1, 
        maxiter = 200,
        plr_maxiter = 4000,
        calculate_elbo = False,
        debug = False,
        display_progress = False,
        epstol = 1e-8):

    n, p = X.shape
    k    = sk.shape[0]
    intercept = np.mean(y)
    y    = y - intercept
    dj   = np.sum(np.square(X), axis = 0)

    outer_elbo_path = list()

    ## Given b and s, update w
    plr1 = PLR(method = 'L-BFGS-B', optimize_b = False, optimize_w = True, optimize_s = False, is_prior_scaled = True,
              debug = debug, display_progress = display_progress, calculate_elbo = calculate_elbo, maxiter = plr_maxiter)
    plr1.fit(X, y, sk, binit = binit, winit = winit, s2init = s2init)

    ## Given s and w, update b.
    plr2 = PLR(method = 'L-BFGS-B', optimize_b = True, optimize_w = False, optimize_s = False, is_prior_scaled = True,
               debug = debug, display_progress = display_progress, calculate_elbo = calculate_elbo, maxiter = plr_maxiter)
    plr2.fit(X, y, sk, binit = plr1.theta, winit = plr1.prior, s2init = plr1.residual_var, unshrink_binit = False)
    
    ## Given b and w, update s
    plr3 = PLR(method = 'L-BFGS-B', optimize_b = False, optimize_w = False, optimize_s = True, is_prior_scaled = True,
               debug = debug, display_progress = display_progress, calculate_elbo = calculate_elbo, maxiter = plr_maxiter)
    plr3.fit(X, y, sk, binit = plr2.theta, winit = plr2.prior, s2init = plr2.residual_var, unshrink_binit = False)

    ## Finally, update s
    plr4 = PLR(method = 'L-BFGS-B', optimize_b = True, optimize_w = True, optimize_s = True, is_prior_scaled = True,
               debug = debug, display_progress = display_progress, calculate_elbo = calculate_elbo, maxiter = plr_maxiter)
    plr4.fit(X, y, sk, binit = plr3.theta, winit = plr3.prior, s2init = plr3.residual_var, unshrink_binit = False)


    obj_path  = plr1.obj_path + plr2.obj_path + plr3.obj_path + plr4.obj_path
    elbo_path = list()
    if calculate_elbo:
        elbo_path = plr1.elbo_path + plr2.elbo_path + plr3.elbo_path + plr4.elbo_path

    res = ResInfo(theta = plr4.theta,
                  coef = plr4.coef,
                  prior = plr4.prior,
                  residual_var = plr4.residual_var,
                  intercept = intercept,
                  elbo_path = elbo_path,
                  obj_path = obj_path,
                  niter = len(obj_path),
                  plr1 = plr1,
                  plr2 = plr2, 
                  plr3 = plr3,
                  plr4 = plr4)

    return res


def method_oneup(X, y, sk, winit, binit = None, s2init = 1, 
        maxiter = 200,
        plr_maxiter = 4000,
        calculate_elbo = False,
        debug = False,
        display_progress = False,
        epstol = 1e-8):

    n, p = X.shape
    k    = sk.shape[0]
    intercept = np.mean(y)
    y    = y - intercept
    dj   = np.sum(np.square(X), axis = 0)

    outer_elbo_path = list()

    ## Given b and s, update w
    plr1 = PLR(method = 'L-BFGS-B', optimize_b = False, optimize_w = True, optimize_s = False, is_prior_scaled = True,
              debug = debug, display_progress = display_progress, calculate_elbo = calculate_elbo, maxiter = plr_maxiter)
    plr1.fit(X, y, sk, binit = binit, winit = winit, s2init = s2init)

    ## Given s, update b and w.
    plr2 = PLR(method = 'L-BFGS-B', optimize_b = True, optimize_w = True, optimize_s = False, is_prior_scaled = True,
               debug = debug, display_progress = display_progress, calculate_elbo = calculate_elbo, maxiter = plr_maxiter)
    plr2.fit(X, y, sk, binit = plr1.theta, winit = plr1.prior, s2init = plr1.residual_var, unshrink_binit = False)
    
    ## Given b and w, update s
    plr3 = PLR(method = 'L-BFGS-B', optimize_b = False, optimize_w = False, optimize_s = True, is_prior_scaled = True,
               debug = debug, display_progress = display_progress, calculate_elbo = calculate_elbo, maxiter = plr_maxiter)
    plr3.fit(X, y, sk, binit = plr2.theta, winit = plr2.prior, s2init = plr2.residual_var, unshrink_binit = False)

    ## Finally, update b, s and w
    plr4 = PLR(method = 'L-BFGS-B', optimize_b = True, optimize_w = True, optimize_s = True, is_prior_scaled = True,
               debug = debug, display_progress = display_progress, calculate_elbo = calculate_elbo, maxiter = plr_maxiter)
    plr4.fit(X, y, sk, binit = plr3.theta, winit = plr3.prior, s2init = plr3.residual_var, unshrink_binit = False)


    obj_path  = plr1.obj_path + plr2.obj_path + plr3.obj_path + plr4.obj_path
    elbo_path = list()
    if calculate_elbo:
        elbo_path = plr1.elbo_path + plr2.elbo_path + plr3.elbo_path + plr4.elbo_path

    res = ResInfo(theta = plr4.theta,
                  coef = plr4.coef,
                  prior = plr4.prior,
                  residual_var = plr4.residual_var,
                  intercept = intercept,
                  elbo_path = elbo_path,
                  obj_path = obj_path,
                  niter = len(obj_path),
                  plr1 = plr1,
                  plr2 = plr2, 
                  plr3 = plr3,
                  plr4 = plr4)

    return res
