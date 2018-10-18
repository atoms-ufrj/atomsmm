"""
.. module:: integrators
   :platform: Unix, Windows
   :synopsis: a module for defining integrator classes.

.. moduleauthor:: Charlles R. A. Abreu <abreu@eq.ufrj.br>

"""

import openmmtools.integrators as openmmtools
from simtk import openmm

from atomsmm.propagators import Propagator as DummyPropagator


class Integrator(openmm.CustomIntegrator, openmmtools.PrettyPrintableIntegrator):
    def __init__(self, stepSize):
        super(Integrator, self).__init__(stepSize)
        self.addGlobalVariable("mvv", 0.0)
        self.obsoleteKinetic = True
        self.obsoleteContextState = True

    def __str__(self):
        return self.pretty_format()

    def addUpdateContextState(self):
        if self.obsoleteContextState:
            super(Integrator, self).addUpdateContextState()
            self.obsoleteContextState = False

    def addComputePerDof(self, variable, expression):
        self.obsoleteKinetic = variable == "v"
        super(Integrator, self).addComputePerDof(variable, expression)

    def addComputeKinetic(self):
        if self.obsoleteKinetic:
            self.addComputeSum("mvv", "m*v*v")
            self.obsoleteKinetic = False

class GlobalThermostatIntegrator(Integrator):
    """
    This class extends OpenMM's CustomIntegrator_ class in order to facilitate the construction
    of NVT integrators which include a global thermostat, that is, one that acts equally and
    simultaneously on all degrees of freedom of the system. In this case, a complete NVT step is
    split as:

    .. math::
        e^{\\delta t \\, iL_\\mathrm{NVT}} = e^{\\frac{1}{2} \\delta t \\, iL_\\mathrm{T}}
                                             e^{\\delta t \\, iL_\\mathrm{NVE}}
                                             e^{\\frac{1}{2} \\delta t \\, iL_\\mathrm{T}}

    The propagator :math:`e^{\\delta t \\, iL_\\mathrm{NVE}}` is a Hamiltonian


    corresponds to a Hamiltonian  :math:`iL_\\mathrm{T}`

    .. _CustomIntegrator: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.CustomIntegrator.html

    Parameters
    ----------
        stepSize : unit.Quantity
            The step size with which to integrate the system (in time unit).
        nveIntegrator : :class:`HamiltonianPropagator`
            The Hamiltonian propagator.
        thermostat : :class:`ThermostatPropagator`, optional, default=DummyPropagator()
            The thermostat propagator.
        randomSeed : int, optional, default=None
            A seed for random numbers.

    """
    def __init__(self, stepSize, nveIntegrator, thermostat=DummyPropagator(), randomSeed=None):
        super(GlobalThermostatIntegrator, self).__init__(stepSize)
        if randomSeed is not None:
            self.setRandomNumberSeed(randomSeed)
        for propagator in [nveIntegrator, thermostat]:
            propagator.addVariables(self)
        thermostat.addSteps(self, 1/2)
        nveIntegrator.addSteps(self)
        thermostat.addSteps(self, 1/2)
