#!/bin/bash

make clean
make test1
make clean_modules
./test1

# gprof
# make clean
# make gprofile
# ./gprofile
# gprof ./gprofile | python gprof2dot.py | dot -Tpng -o output.png
# eog output.png
