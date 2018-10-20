"""
.. module:: integrators
   :platform: Unix, Windows
   :synopsis: a module for defining integrator classes.

.. moduleauthor:: Charlles R. A. Abreu <abreu@eq.ufrj.br>

"""

import math
import random

import openmmtools.integrators as openmmtools
from simtk import openmm
from simtk import unit
from sympy import Symbol
from sympy.parsing.sympy_parser import parse_expr

import atomsmm.propagators as propagators
from atomsmm.propagators import Propagator as DummyPropagator
from atomsmm.utils import InputError


class Integrator(openmm.CustomIntegrator, openmmtools.PrettyPrintableIntegrator):
    def __init__(self, stepSize):
        super(Integrator, self).__init__(stepSize)
        self.addGlobalVariable("mvv", 0.0)
        self.obsoleteKinetic = True
        self.obsoleteContextState = True

    def __str__(self):
        return self.pretty_format()

    def _requirements(self, variable, expression):
        definitions = ("{}={}".format(variable, expression)).split(";")
        names = set()
        symbols = set()
        for definition in definitions:
            name, expr = definition.split("=")
            names.add(Symbol(name.strip()))
            symbols |= parse_expr(expr.replace("^", "**")).free_symbols
        requirements = symbols - names
        return list(str(element) for element in requirements)

    def _checkUpdate(self, variable, expression):
        requirements = self._requirements(variable, expression)
        if self.obsoleteKinetic and "mvv" in requirements:
            super(Integrator, self).addComputeSum("mvv", "m*v*v")
            self.obsoleteKinetic = False

    def _adjustVelocities(self, velocities, masses, targetTwoK):
        mtotal = sum(masses)
        ptotal = sum([m*v for (m, v) in zip(masses, velocities)], openmm.Vec3(0.0, 0.0, 0.0))
        vcm = ptotal/mtotal
        for i in range(len(velocities)):
            velocities[i] -= vcm
        twoK = sum(m*(v[0]**2 + v[1]**2 + v[2]**2) for (m, v) in zip(masses, velocities))
        for i in range(len(velocities)):
            velocities[i] *= math.sqrt(targetTwoK/twoK)

    def addUpdateContextState(self):
        if self.obsoleteContextState:
            super(Integrator, self).addUpdateContextState()
            self.obsoleteContextState = False

    def addComputeGlobal(self, variable, expression):
        if variable == "mvv":
            raise InputError("Cannot assign value to global variable mvv")
        self._checkUpdate(variable, expression)
        super(Integrator, self).addComputeGlobal(variable, expression)

    def addComputePerDof(self, variable, expression):
        self._checkUpdate(variable, expression)
        super(Integrator, self).addComputePerDof(variable, expression)
        if variable == "v":
            self.obsoleteKinetic = True

    def initializeVelocities(self, context, temperature, seed=None):
        if seed is not None:
            random.seed(seed)
        energy_unit = unit.dalton*(unit.nanometer/unit.picosecond)**2
        kT = unit.BOLTZMANN_CONSTANT_kB*unit.AVOGADRO_CONSTANT_NA*temperature/energy_unit
        system = context.getSystem()
        N = system.getNumParticles()
        masses = self.masses = [system.getParticleMass(i)/unit.dalton for i in range(N)]
        velocities = list()
        for m in masses:
            sigma = math.sqrt(kT/m)
            v = sigma*openmm.Vec3(random.gauss(0, 1), random.gauss(0, 1), random.gauss(0, 1))
            velocities.append(v)
        self._adjustVelocities(velocities, masses, 3*N*kT)
        context.setVelocities(velocities)


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


class SIN_R_Integrator(Integrator):
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
    def __init__(self, stepSize, temperature, timeConstant, seed=None):
        super(SIN_R_Integrator, self).__init__(stepSize)
        self.seed = seed
        if seed is not None:
            self.setRandomNumberSeed(seed)
        translation = propagators.TranslationPropagator()
        isoF = propagators.SIN_R_Isokinetic_F_Propagator(temperature)
        isoN = propagators.SIN_R_Isokinetic_N_Propagator(temperature, timeConstant)
        OU = propagators.SIN_R_ForcedOrnsteinUhlenbeckPropagator(temperature, timeConstant, 1/timeConstant)
        propagator = propagators.TrotterSuzukiPropagator(OU, isoN)
        propagator = propagators.TrotterSuzukiPropagator(propagator, translation)
        propagator = propagators.TrotterSuzukiPropagator(propagator, isoF)
        propagator.addVariables(self)
        propagator.addSteps(self)

    def initializeVelocities(self, context, temperature, seed=None):
        super().initializeVelocities(context, temperature/4, seed)
        energy_unit = unit.dalton*(unit.nanometer/unit.picosecond)**2
        kT = unit.BOLTZMANN_CONSTANT_kB*unit.AVOGADRO_CONSTANT_NA*temperature/energy_unit
        state = context.getState(getVelocities=True)
        v = state.getVelocities()*unit.picosecond/unit.nanometer
        Q1 = self.getGlobalVariableByName("Q1")
        Q2 = self.getGlobalVariableByName("Q2")
        v1 = self.getPerDofVariableByName("v1")
        v2 = self.getPerDofVariableByName("v2")
        sigma1 = math.sqrt(kT/Q1)
        sigma2 = math.sqrt(kT/Q2)
        for (i, m) in enumerate(self.masses):
            v1[i] = sigma1*openmm.Vec3(random.gauss(0, 1), random.gauss(0, 1), random.gauss(0, 1))
            v2[i] = sigma2*openmm.Vec3(random.gauss(0, 1), random.gauss(0, 1), random.gauss(0, 1))
            factor = [math.sqrt(kT/(m*x**2 + 0.5*Q1*y**2)) for (x, y) in zip(v[i], v1[i])]
            v[i] = openmm.Vec3(*[f*x for (f, x) in zip(factor, v[i])])
            v1[i] = openmm.Vec3(*[f*x for (f, x) in zip(factor, v1[i])])
        context.setVelocities(v)
        self.setPerDofVariableByName("v1", v1)
        self.setPerDofVariableByName("v2", v2)

    def check(self, context):
        Q1 = self.getGlobalVariableByName("Q1")
        v1s = self.getPerDofVariableByName("v1")
        state = context.getState(getVelocities=True)
        vs = state.getVelocities()*unit.picosecond/unit.nanometer
        for (m, v, v1) in zip(self.masses, vs, v1s):
            print([m*x**2 + 0.5*Q1*y**2 for (x, y) in zip(v, v1)])
