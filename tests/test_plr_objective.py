import unittest
import numpy as np
np.random.seed(200)

from mrashpen.models.plr_ash import PenalizedMrASH as PenMrASH
from mrashpen.models.normal_means_ash import NormalMeansASH
from mrashpen.utils.logs import MyLogger

mlogger = MyLogger(__name__)

class TestPLRObjective(unittest.TestCase):

    def _ash_data(self, n = 100, p = 200, p_causal = 5, pve = 0.5, rho = 0.0, k = 6):

        def sd_from_pve (X, b, pve):
            return np.sqrt(np.var(np.dot(X, b)) * (1 - pve) / pve)

        '''
        ASH prior
        '''
        wk = np.zeros(k)
        wk[1:(k-1)] = np.repeat(1/(k-1), (k - 2))
        wk[k-1] = 1 - np.sum(wk)
        sk = np.arange(k)
        '''
        Equicorr predictors
        X is sampled from a multivariate normal, with covariance matrix V.
        V has unit diagonal entries and constant off-diagonal entries rho.
        '''
        iidX    = np.random.normal(size = n * p).reshape(n, p)
        comR    = np.random.normal(size = n).reshape(n, 1)
        X       = comR * np.sqrt(rho) + iidX * np.sqrt(1 - rho)
        bidx    = np.random.choice(p, p_causal, replace = False)
        b       = np.zeros(p)
        b[bidx] = np.random.normal(size = p_causal)
        sigma   = sd_from_pve(X, b, pve)
        y       = np.dot(X, b) + sigma * np.random.normal(size = n)
        return X, y, b, sigma, wk, sk


    def test_objective_function_deriv(self, eps = 1e-8):
        X, y, b, s, wk, sk = self._ash_data()
        pmash = PenMrASH(X, y, b, s, wk, sk)
        obj   = pmash.objective
        bgrad, wgrad, s2grad = pmash.gradients
        bgrad_numeric = np.zeros(X.shape[1])
        for i in range(X.shape[1]):
            bnew = b.copy()
            bnew[i] += eps
            pmash_beps = PenMrASH(X, y, bnew, s, wk, sk, debug = False)
            bgrad_numeric[i] = (pmash_beps.objective - obj) / eps
        wgrad_numeric = np.zeros(wk.shape[0])
        for i in range(wk.shape[0]):
            wknew = wk.copy()
            wknew[i] += eps
            pmash_weps = PenMrASH(X, y, b, s, wknew, sk, debug = False)
            wgrad_numeric[i] = (pmash_weps.objective - obj) / eps
        pmash_s2eps = PenMrASH(X, y, b, s, wk, sk, debug = False)
        pmash_s2eps.set_s2_eps(eps)
        s2grad_numeric = (pmash_s2eps.objective - obj) / eps
        mlogger.debug(f"Gradient with respect to sigma^2: analytic {s2grad}, numeric {s2grad_numeric}")
        wgrad_string = ','.join([f"{x}" for x in wgrad])
        wgrad_numeric_string = ','.join([f"{x}" for x in wgrad_numeric])
        mlogger.debug(f"Gradient with respect to w_k:")
        mlogger.debug(f"analytic {wgrad_string}")
        mlogger.debug(f"numeric {wgrad_numeric_string}")
        self.assertTrue(np.allclose(bgrad, bgrad_numeric, atol = 1e-4, rtol = 1e-8),
            msg = "Objective function gradient with respect to b does not match numeric results")
        self.assertTrue(np.allclose(wgrad, wgrad_numeric, atol = 1e-2, rtol = 1e-8), 
            msg = "Objective function gradient with respect to w_k does not match numeric results")
        self.assertAlmostEqual(s2grad, s2grad_numeric, places = 1,
            msg = "Objective function gradient with respect to sigma^2 does not match numeric results")
        return


if __name__ == '__main__':
    unittest.main()


