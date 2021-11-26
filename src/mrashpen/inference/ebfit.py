import numpy as np
import collections
from .penalized_regression import PenalizedRegression as PLR
from . import elbo as elbo_py
from ..models.normal_means_ash_scaled import NormalMeansASHScaled


RES_FIELDS = ['theta', 'coef', 'prior', 'residual_var', 'intercept', 'elbo_path', 'outer_elbo_path', 'obj_path', 'niter']
class ResInfo(collections.namedtuple('_ResInfo', RES_FIELDS)):
    __slots__ = ()


def ws_one_step(X, y, b, winit, s2init, sk, dj):
    n, p = X.shape

    ### calculate posterior
    r      = y - np.dot(X, b)
    btilde = b + np.dot(X.T, r) / dj
    nmash  = NormalMeansASHScaled(btilde, np.sqrt(s2init), winit, sk, d = dj, debug = False)
    phijk, mujk, varjk = nmash.posterior()

    ### Update w
    wnew   = np.sum(phijk, axis = 0) / p

    ### Update s2
    bbar   = np.sum(phijk * mujk, axis = 1)
    a1     = np.sum(dj * bbar * btilde)
    varobj = np.dot(r, r) - np.dot(np.square(b), dj) + a1
    s2new  = (varobj + p * (1 - wnew[0]) * s2init) / (n + p * (1 - wnew[0]))

    ### Update ELBO
    elbo   = elbo_py.scalemix(X, y, sk, bbar, wnew, s2new,
                              dj = dj, phijk = phijk, mujk = mujk, varjk = varjk, eps = 1e-8)

    return bbar, wnew, s2new, elbo


def ebfit(X, y, sk, 
          binit = None, winit = None, s2init = None,
          maxiter = 400, qb_maxiter = 50,
          calculate_elbo = True,
          epstol = 1e-8,
          unshrink_method = 'heuristic',
          is_prior_scaled = True,
          display_progress = False,
          debug = False,
          plr_debug = False
         ):
    
    n, p = X.shape
    k    = sk.shape[0]
    intercept = np.mean(y)
    y    = y - intercept
    dj   = np.sum(np.square(X), axis = 0)

    niter = 0
    wk    = winit
    s2    = s2init
    bbar  = binit
    elbo  = np.inf
    theta = np.zeros(p)
    elbo_path       = list()
    obj_path        = list()
    outer_elbo_path = list()
    
    plr = PLR(method = 'L-BFGS-B', optimize_w = False, optimize_s = False,
          debug = debug, 
          display_progress = display_progress, 
          calculate_elbo = calculate_elbo, 
          maxiter = qb_maxiter,
          unshrink_method = unshrink_method,
          prior_optim_method = 'softmax',
          call_from_em = True)
    
    
    for itr in range(maxiter):
        '''
        New coefficients
        '''
        is_step_one = True if itr == 0 else False
        bold = binit if is_step_one else theta
        plr.fit(X, y, sk, binit = bold, winit = wk, s2init = s2, inv_binit = theta, is_binit_coef = is_step_one)
        theta = plr.theta
        if calculate_elbo:
            elbo_path += plr.elbo_path
        obj_path  += plr.obj_path
        '''
        Empirical Bayes update for wk and s2, also advances coef one step
        but we drop that advance
        '''
        bbar, wk, s2, elbo = ws_one_step(X, y, plr.coef, plr.prior, plr.residual_var, sk, dj)
        outer_elbo_path.append(elbo)
        '''
        Termination criteria
        '''
        if (itr > 0) and (elboold - elbo < epstol): break
        elboold = elbo.copy()
    print (f"mr.ash.pen (EM) terminated at iteration {itr + 1}.")

        
    res = ResInfo(theta = theta,
                  coef = bbar,
                  prior = wk,
                  residual_var = s2,
                  intercept = intercept,
                  elbo_path = elbo_path,
                  outer_elbo_path = outer_elbo_path,
                  obj_path = obj_path,
                  niter = len(obj_path))

    return res



def ebfit_old(X, y, sk, wk, binit = None, s2init = 1, 
          maxiter = 1000, qb_maxiter = 100,
          calculate_elbo = True,
          epstol = 1e-8):
    n, p = X.shape
    k    = sk.shape[0]
    intercept = np.mean(y)
    y    = y - intercept
    dj   = np.sum(np.square(X), axis = 0)

    niter = 0
    w  = wk
    s2 = s2init
    b  = binit
    elbo_path = list()
    obj_path  = list()
    elbo  = np.inf
    outer_elbo_path = list()

    for it in range(maxiter):

        ### Remember old parameters
        bold  = b.copy() if b is not None else b
        wold  = w.copy()
        s2old = s2
        elboold = elbo


        ### Update b
        plr = PLR(method = 'L-BFGS-B', optimize_w = False, optimize_s = False, is_prior_scaled = True,
                  debug = False, display_progress = False, calculate_elbo = calculate_elbo, maxiter = qb_maxiter,
                  call_from_em = True, unshrink_method = 'heuristic', prior_optim_method = 'mixsqp')
        plr.fit(X, y, sk, binit = bold, winit = wold, s2init = s2old)
        b = plr.coef
        theta = plr.theta
        r = y - np.dot(X, b)
        elbo_path += plr.elbo_path
        obj_path  += plr.obj_path

        ### calculate ELBO before updating w and s2
        btilde = b + np.dot(X.T, r) / dj
        nmash = NormalMeansASHScaled(btilde, np.sqrt(s2), w, sk, d = dj, debug = False)
        phijk, mujk, varjk = nmash.posterior()
        #elbo   = elbo_py.scalemix(X, y, sk, b, w, s2,
        #                          dj = dj, phijk = phijk, mujk = mujk, varjk = varjk, eps = 1e-8)
        #outer_elbo_path.append(elbo)

        ### Update w
        w = np.sum(phijk, axis = 0) / p

        ### Update s2
        bbar   = np.sum(phijk * mujk, axis = 1)
        a1     = np.sum(dj * bbar * btilde)
        varobj = np.dot(r, r) - np.dot(np.square(b), dj) + a1
        s2     = (varobj + p * (1 - w[0]) * s2old) / (n + p * (1 - w[0]))

        ### Update ELBO / new b
        b      = bbar.copy()
        elbo   = elbo_py.scalemix(X, y, sk, b, w, s2,
                                  dj = dj, phijk = phijk, mujk = mujk, varjk = varjk, eps = 1e-8)
        outer_elbo_path.append(elbo)

        ### Convergence
        ### No elbo in history before one iteration so cannot compare
        if (it > 0) and (elboold - elbo < epstol):
            break
    print (f"mr.ash.pen (EM) terminated at iteration {it + 1}.")

    res = ResInfo(theta = theta,
                  coef = b,
                  prior = w,
                  residual_var = s2,
                  intercept = intercept,
                  elbo_path = elbo_path,
                  outer_elbo_path = outer_elbo_path,
                  obj_path = obj_path,
                  niter = len(obj_path))

    return res
