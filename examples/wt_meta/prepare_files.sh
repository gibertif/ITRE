#!/bin/bash

python ../../tools/clean_file.py -f HILLS -c 1 2 -o COLVARS -d 1
python ../../tools/clean_file.py -f HILLS -c 3 4 -o SIGMAS -d 1
python ../../tools/clean_file.py -f HILLS -c 5 -o HEIGHTS -d 1
