FC = gfortran

FFLAGS = -O -Wall -fbounds-check -g -Wno-uninitialized
LINALGFLAGS = -llapack -lblas

TESTFILE1 = test1.f90
DRIVER    = lbfgsb_driver.f90
LBFGSB    = timer.f linpack.f lbfgsb.f
PRECISION = env_precision.f90
CONSTANTS = global_parameters.f90
FORTUTILS = futils.f90
NMASH_SC  = normal_means_ash_scaled.f90
MRASHPEN  = plr_mrash.f90

all : test1 

test1 : $(PRECISION) $(CONSTANTS) $(FORTUTILS) $(NMASH_SC) $(MRASHPEN) $(LBFGSB) $(DRIVER) $(TESTFILE1)
	$(FC) $(FFLAGS) $? -o $@ $(LINALGFLAGS)

.PHONY : clean clean_modules clean_targets

clean: clean_modules clean_targets

clean_modules:
	rm -f *.mod

clean_targets:
	rm -f test1

