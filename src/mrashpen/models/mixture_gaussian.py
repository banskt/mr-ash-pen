import numpy as np
import scipy.stats

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

def logpdf(X, sk, xzero = 1e-4):
    p = X.shape[0]
    k = sk.shape[0]
    delta_logprob_nonzeroX = np.repeat(-5e23, p) # scipy.stats.norm(0, 1e-12).logpdf(1)
    delta_logprob_zeroX = np.repeat(26.7121, p)  # scipy.stats.norm(0, 1e-12).logpdf(0)
    nonzero_Xmask = np.array(np.abs(X) >= xzero, dtype = bool)
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

def em_one_step(X, sk, wk, xzero = 1e-4):
    p = X.shape[0]
    k = wk.shape[0]
    nonzero_widx  = np.where(wk != 0)[0]
    logLjk = logpdf(X, sk, xzero = xzero)
    wknew = np.zeros(k)
    logwkLjk = logLjk[:, nonzero_widx] + np.log(wk[nonzero_widx])
    log_partition_func = log_sum_exponent(logwkLjk, axis = 1).reshape(-1, 1)
    log_nonzeroFjk = logwkLjk - log_partition_func
    sum_Fjk = log_sum_exponent(log_nonzeroFjk, axis = 0)
    wknew[nonzero_widx] = np.exp(sum_Fjk) / p 
    return wknew


def emfit(X, sk, wk = None, 
          max_iter = 1000, tol = 1e-3,
          xzero = 1e-4):
    k = sk.shape[0]
    if wk is None: wk = np.repeat(1/k, k)
    for itr in range(max_iter):
        wknew = em_one_step(X, sk, wk, xzero = xzero) 
        diff = np.sum(np.abs(wk - wknew))
        wk = wknew
        if diff <= tol:
            break
    return wk

