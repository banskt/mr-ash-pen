import numpy as np
import collections
from .penalized_regression import PenalizedRegression as PLR
from . import elbo as elbo_py
from . import coordinate_descent_step as cd_step
from ..models.normal_means_ash_scaled import NormalMeansASHScaled
from ..models.plr_ash import PenalizedMrASH


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

    ## Update b and w (fixed s)
    plr1 = PLR(method = 'L-BFGS-B', optimize_b = True, optimize_w = True, optimize_s = False, is_prior_scaled = True,
              debug = debug, display_progress = display_progress, calculate_elbo = calculate_elbo, maxiter = plr_maxiter)
    plr1.fit(X, y, sk, binit = binit, winit = winit, s2init = s2init)

    ## Update s and w (fixed b)
    plr2 = PLR(method = 'L-BFGS-B', optimize_b = False, optimize_w = True, optimize_s = True, is_prior_scaled = True,
               debug = debug, display_progress = display_progress, calculate_elbo = calculate_elbo, maxiter = plr_maxiter)
    plr2.fit(X, y, sk, binit = plr1.theta, winit = plr1.prior, s2init = plr1.residual_var, unshrink_binit = False)

    ## Update b, w and s
    plr3 = PLR(method = 'L-BFGS-B', optimize_b = True, optimize_w = True, optimize_s = True, is_prior_scaled = True,
               debug = debug, display_progress = display_progress, calculate_elbo = calculate_elbo, maxiter = plr_maxiter)
    plr3.fit(X, y, sk, binit = plr2.theta, winit = plr2.prior, s2init = plr2.residual_var, unshrink_binit = False)

    obj_path  = plr1.obj_path + plr2.obj_path + plr3.obj_path
    elbo_path = list()
    if calculate_elbo:
        elbo_path = plr1.elbo_path + plr2.elbo_path + plr3.elbo_path

    res = ResInfo(theta = plr3.theta,
                  coef = plr3.coef,
                  prior = plr3.prior,
                  residual_var = plr3.residual_var,
                  intercept = intercept,
                  elbo_path = elbo_path,
                  obj_path = obj_path,
                  niter = len(obj_path),
                  plr1 = plr1,
                  plr2 = plr2, 
                  plr3 = plr3,
                  plr4 = None)

    return res


def method_fixseq_simple(X, y, sk, winit, binit = None, s2init = 1, 
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

    ## Update b and w (fixed s)
    plr1 = PLR(method = 'L-BFGS-B', optimize_b = True, optimize_w = True, optimize_s = False, is_prior_scaled = True,
              debug = debug, display_progress = display_progress, calculate_elbo = calculate_elbo, maxiter = plr_maxiter)
    plr1.fit(X, y, sk, binit = binit, winit = winit, s2init = s2init)

    ## Update s (fixed b and w)
    plr2 = PLR(method = 'L-BFGS-B', optimize_b = False, optimize_w = False, optimize_s = True, is_prior_scaled = True,
               debug = debug, display_progress = display_progress, calculate_elbo = calculate_elbo, maxiter = plr_maxiter)
    plr2.fit(X, y, sk, binit = plr1.theta, winit = plr1.prior, s2init = plr1.residual_var, unshrink_binit = False)

    ## Update b, w and s
    plr3 = PLR(method = 'L-BFGS-B', optimize_b = True, optimize_w = True, optimize_s = True, is_prior_scaled = True,
               debug = debug, display_progress = display_progress, calculate_elbo = calculate_elbo, maxiter = plr_maxiter)
    plr3.fit(X, y, sk, binit = plr2.theta, winit = plr2.prior, s2init = plr2.residual_var, unshrink_binit = False)

    obj_path  = plr1.obj_path + plr2.obj_path + plr3.obj_path
    elbo_path = list()
    if calculate_elbo:
        elbo_path = plr1.elbo_path + plr2.elbo_path + plr3.elbo_path

    res = ResInfo(theta = plr3.theta,
                  coef = plr3.coef,
                  prior = plr3.prior,
                  residual_var = plr3.residual_var,
                  intercept = intercept,
                  elbo_path = elbo_path,
                  obj_path = obj_path,
                  niter = len(obj_path),
                  plr1 = plr1,
                  plr2 = plr2, 
                  plr3 = plr3,
                  plr4 = None)

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


def method_oneup(X, y, sk, winit, binit = None, s2init = 1, maxiter = 200,
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


def method_init_gb(X, y, sk, winit, binit = None, s2init = 1, 
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

    wopt  = fit_g_from_binit(binit, sk, winit)
    theta = np.zeros(p)
    pmash = PenalizedMrASH(X, y, theta, np.sqrt(s2init), wopt, sk, dj = dj, debug = debug, is_prior_scaled = True)
    theta_opt = pmash.unshrink_b(binit)

    ## Update b, s and w
    plr1 = PLR(method = 'L-BFGS-B', optimize_b = True, optimize_w = True, optimize_s = False, is_prior_scaled = True,
              debug = debug, display_progress = display_progress, calculate_elbo = calculate_elbo, maxiter = plr_maxiter)
    plr1.fit(X, y, sk, binit = theta_opt, winit = wopt, s2init = s2init, unshrink_binit = False)

    obj_path  = plr1.obj_path
    elbo_path = list()
    if calculate_elbo:
        elbo_path = plr1.elbo_path

    res = ResInfo(theta = plr1.theta,
                  coef = plr1.coef,
                  prior = plr1.prior,
                  residual_var = plr1.residual_var,
                  intercept = intercept,
                  elbo_path = elbo_path,
                  obj_path = obj_path,
                  niter = len(obj_path),
                  plr1 = plr1,
                  plr2 = None, 
                  plr3 = None,
                  plr4 = None)

    return res


''' 
log(sum(exp(z))) = M + log(sum(exp(z - M)))
z is a matrix of shape (p, k)
Sum is over k
'''
def log_sum_exponent(z, axis = 1): 
    zmax = np.max(z, axis = axis)
    if axis == 0:
        sub_zmax = zmax.reshape(1, -1)
    elif axis == 1:
        sub_zmax = zmax.reshape(-1, 1)
    logsum = np.log(np.sum(np.exp(z - sub_zmax), axis = axis)) + zmax
    return logsum

def matrix_lognormpdf(X, sk):
    p = X.shape[0]
    k = sk.shape[0]
    delta_logprob_nonzeroX = np.repeat(-5e23, p) # scipy.stats.norm(0, 1e-12).logpdf(1)
    delta_logprob_zeroX = np.repeat(26.7121, p)  # scipy.stats.norm(0, 1e-12).logpdf(0)
    nonzero_Xmask = np.array(X  != 0, dtype = bool)
    nonzero_smask = np.array(sk != 0, dtype = bool)
    assert (np.sum(~nonzero_smask) <= 1)
    log_prob = np.zeros((p, k))
    if np.sum(~nonzero_smask) == 1:
        log_prob[nonzero_Xmask, ~nonzero_smask] = delta_logprob_nonzeroX[nonzero_Xmask]
        log_prob[~nonzero_Xmask, ~nonzero_smask] = delta_logprob_zeroX[~nonzero_Xmask]
    nzk = np.sum(nonzero_smask)
    nzSK = sk[nonzero_smask].reshape(1, nzk)
    log_prob[:, nonzero_smask]  = - 0.5 * np.square(X.reshape(p, 1) / nzSK)
    log_prob[:, nonzero_smask] += - 0.5 * np.log(2 * np.pi * np.square(nzSK))
    return log_prob

def em_step_mixture_gaussians(X, sk, wk):
    p = X.shape[0]
    k = wk.shape[0]
    nonzero_widx  = np.where(wk != 0)[0]
    nonzero_sidx  = np.where(sk != 0)[0]
    logLjk = matrix_lognormpdf(X, sk)
    wknew = np.zeros(k)
    logwkLjk = logLjk[:, nonzero_widx] + np.log(wk[nonzero_widx])
    log_partition_func = log_sum_exponent(logwkLjk, axis = 1).reshape(-1, 1)
    log_nonzeroFjk = logwkLjk - log_partition_func
    sum_Fjk = log_sum_exponent(log_nonzeroFjk, axis = 0)
    wknew[nonzero_widx] = np.exp(sum_Fjk) / p
    return wknew


def fit_g_from_binit(b, sk, wk, max_iter = 1000, tol = 1e-3):
    for itr in range(max_iter):
        wknew = em_step_mixture_gaussians(b, sk, wk)
        diff = np.sum(np.abs(wk - wknew))
        wk = wknew
        if diff <= tol:
            break
    return wk
