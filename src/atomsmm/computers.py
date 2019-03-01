"""
.. module:: computers
   :platform: Unix, Windows
   :synopsis: a module for defining computers, which are subclasses of OpenMM Context_ class.

.. moduleauthor:: Charlles R. A. Abreu <abreu@eq.ufrj.br>

.. _Context: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.Context.html

"""

from simtk import openmm

import atomsmm


class VirialComputer(openmm.Context):
    def __init__(self, system, platform, properties=dict(), **kwargs):
        integrator = openmm.CustomIntegrator(0)
        self._system = atomsmm.ComputingSystem(system, **kwargs)
        super().__init__(self._system, integrator, platform, properties)
