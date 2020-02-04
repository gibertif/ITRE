#!/bin/bash

header='#! FIELDS time d1.x d1.y'
source ~/Programs/plumed2/sourceme.sh
#
#files=`ls cvs_?`
#
echo '' > time.dat
#
#for file in $files
#do
#	echo $header > colvars
#	cat $file >> colvars
#	start=`date +%s`
#	plumed driver --noatoms --plumed plumed_reweight.dat
#	end=`date +%s`
#	runtime=$((end-start))
#	echo $runtime >> time.dat
#
#done
#

files=`ls cvs_40`

for file in $files
do
	echo $header > colvars
	cat $file >> colvars
	start=`date +%s`
	plumed driver --noatoms --plumed plumed_reweight.dat
	end=`date +%s`
	runtime=$((end-start))
	echo $runtime >> time.dat
done
