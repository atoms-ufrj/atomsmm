#!/bin/bash -eE
:<<"::batch"
@echo off
conda install --yes -c omnia openmm
pip install %*
goto :end
::batch
conda install --yes -c omnia openmm
pip install $*
exit $?
:<<"::done"
:end
::done
