'''
Python wrapper for mr.ash.alpha 
'''
import numpy as np
from scipy import optimize as sp_optimize
import logging
import numbers

import os
import tempfile
import subprocess

import rpy2.robjects as robj
import rpy2.robjects.vectors as rvec
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
numpy2ri.activate()
from rpy2.robjects.conversion import localconverter

from ..utils.logs import MyLogger
from ..utils import R_utils

class MrASHR:

    def __init__(self, option = "r2py", debug = False):
        self._option = option
        self.rscript_file = os.path.realpath(os.path.join(os.path.dirname(__file__), "../utils/fit_mrash.R"))
        if debug:
            self.logger = MyLogger(__name__)
        else:
            self.logger = MyLogger(__name__, level = logging.INFO)


    @property
    def coef(self):
        return self._fitdict['beta']


    @property
    def prior(self):
        return self._fitdict['pi']


    @property
    def residual_var(self):
        return self._fitdict['sigma2']


    @property
    def fitobj(self):
        return self._fitdict


    @property
    def obj_path(self):
        _obj_path = self._fitdict['varobj']
        if not (isinstance(_obj_path, list) or isinstance(_obj_path, np.ndarray)):
            _obj_path = list([_obj_path])
        return _obj_path


    @property
    def niter(self):
        return self._fitdict['iter']


    @property
    def intercept(self):
        return self._fitdict['intercept']


    @property
    def elbo_path(self):
        _elbo_path = self._fitdict['varobj']
        if not (isinstance(_elbo_path, list) or isinstance(_elbo_path, np.ndarray)):
            _elbo_path = list([_elbo_path])
        return _elbo_path


    def array_reduce(self, x):
        ndim = x.ndim
        if ndim == 1:
            res = x[0] if x.shape[0] == 1 else x
        elif ndim == 2:
            res = x.reshape(-1) if x.shape[1] == 1 else x
        return res 
    

    def robj2dict_recursive(self, robj):
        res = dict()
        for key in robj.names:
            elem = robj.rx2(key)
            if isinstance(elem, (rvec.FloatVector, rvec.IntVector)):
                res[key] = self.array_reduce(np.array(elem))
            elif isinstance(elem, rvec.StrVector):
                self.logger.error(f"ERROR: Abnormal StrVector output")
            elif isinstance(elem, np.ndarray):
                res[key] = self.array_reduce(elem)
            elif isinstance(elem, rvec.ListVector):
                res[key] = self.robj2dict_recursive(elem)
        return res


    def fit(self, X, y, sk, binit = None, winit = None, s2init = None, 
            epstol = 1e-12, convtol = 1e-8, maxiter = 2000,
            update_pi = True, update_sigma2 = True):
        '''
        Initialization
        '''
        n, p = X.shape
        k = sk.shape[0]
        if binit is None:
            binit = np.zeros(p)
        if winit is None:
            winit = self.initialize_mixcoef(k)
        if s2init is None:
            s2init = 1.0
        assert(np.abs(np.sum(winit) - 1) < 1e-5)
        '''
        Fit with R
        '''
        if self._option == "r2py":
            self._fitdict = self.r2py_wrapper(X, y, sk, binit, winit, s2init, maxiter, epstol, convtol,
                                              update_pi = update_pi, update_sigma2 = update_sigma2)
        elif self._option == "rds":
            self._fitdict = self.rds_wrapper(X, y, sk, binit, winit, s2init, maxiter, epstol, convtol,
                                             update_pi = update_pi, update_sigma2 = update_sigma2)
        return


    def r2py_wrapper(self, X, y, sk, binit, wkinit, s2init, maxiter, epstol, convtol,
                     update_pi = True, update_sigma2 = True):
        mrashR = importr('mr.ash.alpha')
        n, p = X.shape
        r_X      = robj.r.matrix(X, nrow = n, ncol = p)
        r_y      = rvec.FloatVector(y)
        r_sk2    = rvec.FloatVector(np.square(sk))
        r_binit  = rvec.FloatVector(binit)
        r_wkinit = rvec.FloatVector(wkinit)
        r_tol    = rvec.ListVector({'epstol': epstol,
                                'convtol': convtol})

        r_fit = mrashR.mr_ash(r_X, r_y,
                              standardize = False, intercept = True,
                              max_iter = maxiter,
                              sa2 = r_sk2,
                              beta_init = r_binit,
                              pi = r_wkinit,
                              sigma2 = s2init,
                              update_pi = update_pi,
                              update_sigma2 = update_sigma2,
                              tol = r_tol
                             )

        #with localconverter(robj.default_converter):
        #    r_fit_conv = robj.conversion.rpy2py(r_fit)

        fit_dict = self.robj2dict_recursive(r_fit)
        return fit_dict


    def rds_wrapper(self, X, y, sk, binit, wkinit, s2init, maxiter, epstol, convtol,
                    update_pi = True, update_sigma2 = True):
        os_handle, data_rds_file = tempfile.mkstemp(suffix = ".rds")
        datadict = {'X': X, 'y': y, 'sk2': np.square(sk), 
                    'binit': binit, 'winit': wkinit, 's2init': s2init}
        R_utils.save_rds(datadict, data_rds_file)
        os_handle, out_rds_file = tempfile.mkstemp(suffix = ".rds")
        cmd  = ["Rscript",   self.rscript_file]
        cmd += ["--outfile", out_rds_file]
        cmd += ["--infile",  data_rds_file]
        cmd += ["--maxiter", f"{maxiter}"]
        cmd += ["--epstol",  f"{epstol}"]
        cmd += ["--convtol", f"{convtol}"]
        if not update_pi:     cmd += ["--fix_pi"]
        if not update_sigma2: cmd += ["--fix_sigma2"]

        process = subprocess.Popen(cmd,
                                   stdout = subprocess.PIPE,
                                   stderr = subprocess.PIPE
                                  )
        res     = process.communicate()
        self.logger.info(res[0].decode('utf-8'))
    
        if len(res[1].decode('utf-8')) > 0:
            self.logger.debug("ERROR ==>")
            self.logger.debug(res[1].decode('utf-8'))

        retcode  = process.returncode
        fit_dict = R_utils.load_rds(out_rds_file) if retcode == 0 else None
        if os.path.exists(data_rds_file): os.remove(data_rds_file)
        if os.path.exists(out_rds_file):  os.remove(out_rds_file)
        return fit_dict
