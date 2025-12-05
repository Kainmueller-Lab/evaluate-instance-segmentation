#!/bin/bash

if [ -z "$1" ]
then
    echo "No argument supplied"
	exit
fi

pwd=$(pwd)
dn=$(basename ${pwd})

# for tiffiles it should be just
# python evaluate.py <pred-file> <gt-file>

parallel -j10 python evaluate.py ${dn}predictions_${1}_{} data/test/{} segBgDil ${2} ::: $(ls data/test)
