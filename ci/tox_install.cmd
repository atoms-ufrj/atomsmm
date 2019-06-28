#!/bin/bash -eE
:<<"::batch"
@echo off
conda install --prefix=%1 --yes -c omnia openmm
for /f "tokens=1,* delims= " %%a in ("%*") do set ALL_BUT_FIRST=%%b
pip install %ALL_BUT_FIRST%
goto :end
::batch
conda create --prefix=$1 --yes python=3.7
conda install --prefix=$1 --yes -c omnia openmm
conda install --prefix=$1 --yes pytest pytest-cov sphinx sphinx_rtd_theme
conda install --prefix=$1 --yes -c conda-forge future_fstrings sphinxcontrib-bibtex
pip install ${@:2}
exit $?
:<<"::done"
:end
::done
