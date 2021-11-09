import numpy as np
import os
import tempfile
import subprocess

from . import R_utils

rscript_file = os.path.realpath(os.path.join(os.path.dirname(__file__), "fit_lasso.R"))

def fit(X, y, nfolds = 10, alpha = 1.0):
    os_handle, data_rds_file = tempfile.mkstemp(suffix = ".rds")
    datadict = {'X': X, 'y': y}
    R_utils.save_rds(datadict, data_rds_file)
    os_handle, out_rds_file = tempfile.mkstemp(suffix = ".rds")
    cmd  = ["Rscript",   rscript_file]
    cmd += ["--outfile", out_rds_file]
    cmd += ["--infile",  data_rds_file]
    cmd += ["--nfolds", f"{nfolds}"]
    cmd += ["--alpha",  f"{alpha}"]


    process = subprocess.Popen(cmd,
                               stdout = subprocess.PIPE,
                               stderr = subprocess.PIPE
                              )
    res     = process.communicate()
    if len(res[0].decode('utf-8')) > 0:
        print(res[0].decode('utf-8'))
    if len(res[1].decode('utf-8')) > 0:
        print("ERROR ==>")
        print(res[1].decode('utf-8'))
    retcode  = process.returncode
    fit_dict = R_utils.load_rds(out_rds_file) if retcode == 0 else None
    if os.path.exists(data_rds_file): os.remove(data_rds_file)
    if os.path.exists(out_rds_file):  os.remove(out_rds_file)
    intercept = fit_dict['mu']
    coef = fit_dict['beta']
    return intercept, coef, fit_dict
