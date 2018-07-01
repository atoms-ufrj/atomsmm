#!/bin/bash -eE
:<<"::batch"
@echo off
pip install numpy six
pip install %*
goto :end
::batch
pip install numpy six
pip install $*
exit $?
:<<"::done"
:end
::done
