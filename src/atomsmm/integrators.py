"""
.. module:: integrators
   :platform: Unix, Windows
   :synopsis: a module for defining integrator classes.

.. moduleauthor:: Charlles R. A. Abreu <abreu@eq.ufrj.br>

"""

import math
import random
import re

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
        super().__init__(stepSize)
        self.addGlobalVariable("mvv", 0.0)
        self.obsoleteKinetic = True
        self.forceFinder = re.compile("f[0-9]*")
        self.obsoleteContextState = True

    def __str__(self):
        return self.pretty_format()

    def _required_variables(self, variable, expression):
        """
        Returns a list of strings containting the names of all global and per-dof variables
        required by an OpenMM CustomIntegrator operation.

        """
        definitions = ("{}={}".format(variable, expression)).split(";")
        names = set()
        symbols = set()
        for definition in definitions:
            name, expr = definition.split("=")
            names.add(Symbol(name.strip()))
            symbols |= parse_expr(expr.replace("^", "**")).free_symbols
        return list(str(element) for element in (symbols - names))

    def _checkUpdate(self, variable, expression):
        """
        Check whether it is necessary to update the mvv global variable (twice the kinetic energy)
        or to let the forces update the context state.

        """
        requirements = self._required_variables(variable, expression)
        if self.obsoleteKinetic and "mvv" in requirements:
            super(Integrator, self).addComputeSum("mvv", "m*v*v")
            self.obsoleteKinetic = False
        if self.obsoleteContextState and any(self.forceFinder.match(s) for s in requirements):
            super(Integrator, self).addUpdateContextState()
            self.obsoleteContextState = False

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

    def initializeVelocities(self, context, temperature):
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
        mtotal = sum(masses)
        ptotal = sum([m*v for (m, v) in zip(masses, velocities)], openmm.Vec3(0.0, 0.0, 0.0))
        vcm = ptotal/mtotal
        for i in range(len(velocities)):
            velocities[i] -= vcm
        twoK = sum(m*(v[0]**2 + v[1]**2 + v[2]**2) for (m, v) in zip(masses, velocities))
        factor = math.sqrt(3*N*kT/twoK)
        for i in range(len(velocities)):
            velocities[i] *= factor
        context.setVelocities(velocities)

    def setRandomNumberSeed(self, seed):
        super(Integrator, self).setRandomNumberSeed(seed)
        random.seed(seed)


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
    def __init__(self, stepSize, nveIntegrator, thermostat=DummyPropagator()):
        super(GlobalThermostatIntegrator, self).__init__(stepSize)
        for propagator in [nveIntegrator, thermostat]:
            propagator.addVariables(self)
        thermostat.addSteps(self, 1/2)
        nveIntegrator.addSteps(self)
        thermostat.addSteps(self, 1/2)


class SIN_R_Integrator(Integrator):
    """
    This class is an implementation of the Stochastic-Iso-NH-RESPA or SIN(R) method of Leimkuhler,
    Margul, and Tuckerman :cite:`Leimkuhler_2013`. The method consists in solving the following
    equations for each degree of freedom (DOF) in the system:

    .. math::
        & \\frac{dx}{dt} = v \\\\
        & \\frac{dv}{dt} = \\frac{f}{m} - \\lambda v \\\\
        & \\frac{dv_1}{dt} = - \\lambda v_1 - v_2 v_1 \\\\
        & dv_2 = \\frac{Q_1 v_1^2 - kT}{Q_2}dt - \\gamma v_2 dt + \\sqrt{\\frac{2 \\gamma kT}{Q_2}} dW

    where:

    .. math::
        \\lambda = \\frac{f v - \\frac{1}{2} Q_1 v_2 v_1^2}{m v^2 + \\frac{1}{2} Q_1 v_1^2}.

    A consequence of these equations is that

    .. math::
        m v^2 + \\frac{1}{2} Q_1 v_1^2 = kT.

    The equations are integrated by a reversible, multiple timescale numerical scheme.

    Parameters
    ----------
        stepSize : unit.Quantity
            The largest time step for numerically integrating the system of equations.
        loops : list(int)
            A list of `N` integers, where `loops[k]` determines how many steps involving force group
            `k` are internally executed for every step involving force group `k+1`. Thus, group `0`
            is composed of the fastest forces, while group `N-1` is composed of the slowest ones.
        temperature : unit.Quantity
            The temperature to which the configurational sampling should correspond.
        timeScale : unit.Quantity
            A time scale :math:`\\tau` from which the inertial parameters are computed as
            :math:`Q_1 = Q_2 = kT\\tau^2`.
        frictionConstant : unit.Quantity
            The friction constant :math:`\\gamma` present in the stochastic equation of motion for
            per-DOF variable :math:`v_2`.
        scheme : str, optional, default="middle"
            The scheme to be used for splitting the equations of motion. Available optionas are
            `middle` (default), `xi`, `xm`, and `xo`.

    """
    def __init__(self, stepSize, loops, temperature, timeScale, frictionConstant, scheme="middle", nsy=1, nres=1):
        super().__init__(stepSize)
        isoF = propagators.MassiveIsokineticPropagator(temperature, timeScale, forceDependent=True)
        isoN = propagators.MassiveIsokineticPropagator(temperature, timeScale, forceDependent=False)
        force = "Q1*v1*v1 - kT" if scheme == "middle" else None
        OU = propagators.OrnsteinUhlenbeckPropagator(temperature, frictionConstant, "v2", "Q2", force)
        if scheme == "middle":
            central = propagators.TrotterSuzukiPropagator(isoN, OU)
            propagator = propagators.RespaPropagator(loops, core=central, boost=isoF)
        else:
            v2boost = propagators.GenericBoostPropagator("v2", "Q2", "Q1*v1*v1 - kT")
            NH = propagators.TrotterSuzukiPropagator(isoN, v2boost)
            shell = propagators.SuzukiYoshidaPropagator(propagators.SplitPropagator(NH, nres), nsy)
            level = dict(xi=0, xm=1, xo=len(loops)-1)
            propagator = propagators.RespaPropagator(loops, core=OU, shell={level[scheme]: shell}, boost=isoF)
        propagator.addVariables(self)
        propagator.addSteps(self)
        self.requiresInitialization = True

    def step(self, steps):
        if self.requiresInitialization:
            kT = self.getGlobalVariableByName("kT")
            Q1 = self.getGlobalVariableByName("Q1")
            Q2 = self.getGlobalVariableByName("Q2")
            v1 = self.getPerDofVariableByName("v1")
            v2 = self.getPerDofVariableByName("v2")
            S1 = math.sqrt(2*kT/Q1)
            S2 = math.sqrt(kT/Q2)
            for i in range(len(v1)):
                v1[i] = openmm.Vec3(random.gauss(0, S1), random.gauss(0, S1), random.gauss(0, S1))
                v2[i] = openmm.Vec3(random.gauss(0, S2), random.gauss(0, S2), random.gauss(0, S2))
            self.setPerDofVariableByName("v1", v1)
            self.setPerDofVariableByName("v2", v2)
            self.requiresInitialization = False
        return super().step(steps)
