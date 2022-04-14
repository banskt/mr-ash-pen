FC = gfortran

FFLAGS = -O3 -Wall -fbounds-check -g -Wno-uninitialized -fno-automatic -ffast-math
#FFLAGS = -O3 -Wall
# The default, which is -fautomatic, uses the stack for local variables smaller
# than the value given by -fmax-stack-var-size.
LINALGFLAGS = -llapack -lblas

TESTFILE1 = test1.f90
TESTFILE3 = test3.f90
DRIVER    = optutils.f90 lbfgsb_driver.f90
LBFGSB    = timer.f linpack.f lbfgsb.f
PRECISION = env_precision.f90
CONSTANTS = global_parameters.f90
FORTUTILS = futils.f90
NMASH_SC  = normal_means_ash_scaled.f90
MRASHPEN  = plr_mrash.f90

all : test1 test3 

test1 : $(PRECISION) $(CONSTANTS) $(FORTUTILS) $(NMASH_SC) $(MRASHPEN) $(LBFGSB) $(DRIVER) $(TESTFILE1)
	$(FC) $(FFLAGS) $? -o $@ $(LINALGFLAGS)

test3 : $(PRECISION) $(CONSTANTS) $(FORTUTILS) $(NMASH_SC) $(MRASHPEN) $(TESTFILE3)
	$(FC) $(FFLAGS) $? -o $@ $(LINALGFLAGS)

gprofile : $(PRECISION) $(CONSTANTS) $(FORTUTILS) $(NMASH_SC) $(MRASHPEN) $(TESTFILE3)
	$(FC) $(FFLAGS) $? -pg -g -o $@ $(LINALGFLAGS)

.PHONY : clean clean_modules clean_targets

clean: clean_modules clean_targets

clean_modules:
	rm -f *.mod

clean_targets:
	rm -f test1 test3 gprofile
