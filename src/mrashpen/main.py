import numpy as np
import argparse
import sys
import unittest
import mpi4py

from mrashpen.utils.logs import MyLogger
from mrashpen.utils.unittest_tester import UnittestTester
from mrashpen.models.tests.test_normal_means import TestNMAshPy
from mrashpen.models.tests.test_normal_means_scaled import TestNMAshScaledPy
from mrashpen.models.tests.test_plr_objective import TestPLRObjective
from mrashpen.utils import project

mlogger = MyLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Mr.AshPen: penalized regression method for sparse multiple regression using adaptive shrinkage (ASH) prior')
    parser.add_argument('--test',
                        dest = 'test',
                        action = 'store_true',
                        help = 'Perform unit tests')
    parser.add_argument('--version',
                        dest = 'version',
                        action = 'store_true',
                        help = 'Print version number')
    res = parser.parse_args()
    return res


def run_unittests():
    for mtest in [TestNMAshScaledPy, 
                  TestNMAshPy,
                  TestPLRObjective,
                 ]:
        tester = UnittestTester(mtest)
        tester.execute()
        del tester
    return


def main():

    opts = parse_args()
    if opts.test:
        run_unittests()
    elif opts.version:
        print ("Mr.AshPen version {:s}".format(project.version()))
    elif opts.do_lbfgsb:
        run_lbfgsb(opts)

if __name__ == "__main__":
    main()
