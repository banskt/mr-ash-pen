import numpy as np
import collections
from .penalized_regression import PenalizedRegression as PLR
from . import elbo as elbo_py
from ..models.normal_means_ash_scaled import NormalMeansASHScaled


RES_FIELDS = ['theta', 'coef', 'prior', 'residual_var', 'intercept', 'elbo_path', 'outer_elbo_path', 'obj_path']
class ResInfo(collections.namedtuple('_ResInfo', RES_FIELDS)):
    __slots__ = ()


def ebfit(X, y, sk, wk, binit = None, s2init = 1, 
          maxiter = 1000, qb_maxiter = 100,
          epstol = 1e-8):
    n, p = X.shape
    k    = sk.shape[0]
    if binit is None: binit = np.zeros(p)
    intercept = np.mean(y)
    y    = y - intercept
    dj   = np.sum(np.square(X), axis = 0)

    niter = 0
    w  = wk
    s2 = s2init
    b  = binit
    r  = y - np.dot(X, b)
    elbo_path = list()
    obj_path  = list()
    theta = b.copy()
    elbo  = np.inf
    outer_elbo_path = list()

    for it in range(maxiter):

        ### Remember old parameters
        bold  = b.copy()
        wold  = w.copy()
        rold  = r.copy()
        s2old = s2
        thetaold = theta.copy()
        elboold = elbo


        ### Update b
        plr = PLR(method = 'L-BFGS-B', optimize_w = False, optimize_s = False, is_prior_scaled = True,
                  debug = False, display_progress = False, calculate_elbo = True, maxiter = qb_maxiter)
        plr.fit(X, y, sk, binit = theta, winit = w, s2init = s2)
        b = plr.coef
        theta = plr.theta
        r = y - np.dot(X, b)
        elbo_path += plr.elbo_path
        obj_path  += plr.obj_path

        ### calculate ELBO before updating w and s2
        btilde = b + np.dot(X.T, r) / dj
        nmash = NormalMeansASHScaled(btilde, np.sqrt(s2), w, sk, d = dj, debug = False)
        phijk, mujk, varjk = nmash.posterior()
        elbo   = elbo_py.scalemix(X, y, sk, b, w, s2,
                                  dj = dj, phijk = phijk, mujk = mujk, varjk = varjk, eps = 1e-8)
        outer_elbo_path.append(elbo)

        ### Update w
        w = np.sum(phijk, axis = 0) / p

        ### Update s2
        bbar   = np.sum(phijk * mujk, axis = 1)
        a1     = np.sum(dj * bbar * btilde)
        varobj = np.dot(r, r) - np.dot(np.square(b), dj) + a1
        s2     = (varobj + p * (1 - w[0]) * s2old) / (n + p * (1 - w[0]))

        ### Convergence
        ### No elbo in history before one iteration so cannot compare
        if (it > 0) and (elboold - elbo < epstol):
            break

    res = ResInfo(theta = theta,
                  coef = b,
                  prior = w,
                  residual_var = s2,
                  intercept = intercept,
                  elbo_path = elbo_path,
                  outer_elbo_path = outer_elbo_path,
                  obj_path = obj_path)

    return res
