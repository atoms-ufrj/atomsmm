"""
.. module:: integrators
   :platform: Unix, Windows
   :synopsis: a module for defining integrator classes.

.. moduleauthor:: Charlles R. A. Abreu <abreu@eq.ufrj.br>

"""

from simtk import openmm

from atomsmm.algorithms import DummyThermostat


class GlobalThermostatIntegrator(openmm.CustomIntegrator):
    def __init__(self, stepSize, nveIntegrator, thermostat=DummyThermostat(), randomSeed=None):
        super(GlobalThermostatIntegrator, self).__init__(stepSize)
        if randomSeed is not None:
            self.setRandomNumberSeed(randomSeed)
        for algorithm in [nveIntegrator, thermostat]:
            algorithm.addVariables(self)
        thermostat.addSteps(self, 1/2)
        nveIntegrator.addSteps(self)
        thermostat.addSteps(self, 1/2)
