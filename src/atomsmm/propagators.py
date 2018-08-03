"""
.. module:: propagators
   :platform: Unix, Windows
   :synopsis: a module for defining propagator classes.

.. moduleauthor:: Charlles R. A. Abreu <abreu@eq.ufrj.br>

.. _CustomIntegrator: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.CustomIntegrator.html
.. _VerletIntegrator: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.VerletIntegrator.html

"""

import math
from copy import deepcopy

from simtk import openmm
from simtk import unit


class Propagator:
    """
    This is the base class for propagators, which are building blocks for
    constructing CustomIntegrator_ objects in OpenMM. Shortly, a propagator translates the effect
    of an exponential operator like
    :math:`e^{\\delta t \\, iL}`.
    This effect can be either the exact solution of a system of deterministic or Stochastic
    differential equations or an approximate solution obtained by a splitting scheme such as
    :math:`e^{\\delta t \\, iL} \\approx e^{\\delta t \\, iL_A} e^{\\delta t \\, iL_B}`,
    for instance.

    """
    def __init__(self):
        self.globalVariables = dict()
        self.perDofVariables = dict()
        self.persistent = list()

    def declareVariables(self):
        pass

    def addVariables(self, integrator):
        for (name, value) in self.globalVariables.items():
            integrator.addGlobalVariable(name, value)
        for (name, value) in self.perDofVariables.items():
            integrator.addPerDofVariable(name, value)

    def addSteps(self, integrator, fraction=1.0):
        pass

    def integrator(self, stepSize=1.0*unit.femtosecond):
        """
        This method generates an OpenMM CustomIntegrator_ object which implements the effect of the
        propagator.

        Parameters
        ----------
            stepSize : unit.Quantity, optional, default=1.0*unit.femtosecond
                The step size with which to integrate the system (in time units).

        Returns
        -------
            openmm.CustomIntegrator

        """
        integrator = openmm.CustomIntegrator(stepSize)
        self.addVariables(integrator)
        self.addSteps(integrator)
        return integrator


class TrotterSuzukiPropagator(Propagator):
    """
    This class combines two propagators :math:`A = e^{\\delta t \\, iL_A}` and
    :math:`B = e^{\\delta t \\, iL_B}` by using the time-symmetric Trotter-Suzuki splitting scheme
    :cite:`Suzuki_1976` :math:`C = B^{1/2} A B^{1/2}`, that is,

    .. math::
        e^{\\delta t \\, iL_C} = e^{{1/2} \\delta t \\, iL_B}
                                 e^{\\delta t \\, iL_A}
                                 e^{{1/2} \\delta t \\, iL_B}.

    .. note::
        It is possible to create nested Trotter-Suzuki propagators. If, for instance, :math:`B` is
        a Trotter-Suzuki propagator given by :math:`E^{1/2} D E^{1/2}`, then an object instantiated
        by `TrotterSuzukiPropagator(A,B)` will be a propagator corresponding to
        :math:`E^{1/4} D^{1/2} E^{1/4} A E^{1/4} D^{1/2} E^{1/4}`.

    """
    def __init__(self, A, B):
        super(TrotterSuzukiPropagator, self).__init__()
        self.A = deepcopy(A)
        self.B = deepcopy(B)
        for propagator in [self.A, self.B]:
            self.globalVariables.update(propagator.globalVariables)
            self.perDofVariables.update(propagator.perDofVariables)

    def addSteps(self, integrator, fraction=1.0):
        self.B.addSteps(integrator, 0.5*fraction)
        self.A.addSteps(integrator, fraction)
        self.B.addSteps(integrator, 0.5*fraction)


class VelocityVerletPropagator(Propagator):
    """
    This class implements a simple Verlocity Verlet propagator.

    .. math::
        e^{\\delta t \\, iL_\\mathrm{NVE}} = e^{\\frac{1}{2} \\delta t \\mathbf{F}^T \\nabla_\\mathbf{p}}
                                             e^{\\delta t \\mathbf{p}^T \\mathbf{M}^{-1} \\nabla_\\mathbf{r}}
                                             e^{\\frac{1}{2} \\delta t \\mathbf{F}^T \\nabla_\\mathbf{p}}

    .. note::
        In the original OpenMM VerletIntegrator_ class, the implemented propagator is a leap-frog
        version of the Verlet method.

    """
    def __init__(self):
        super(VelocityVerletPropagator, self).__init__()
        self.declareVariables()

    def declareVariables(self):
        self.perDofVariables["x0"] = 0
        self.persistent = None

    def addSteps(self, integrator, fraction=1.0):
        Dt = "; Dt=%s*dt" % fraction
        integrator.addUpdateContextState()
        integrator.addComputePerDof("v", "v+0.5*Dt*f/m" + Dt)
        integrator.addComputePerDof("x0", "x")
        integrator.addComputePerDof("x", "x+Dt*v" + Dt)
        integrator.addConstrainPositions()
        integrator.addComputePerDof("v", "(x-x0)/Dt+0.5*Dt*f/m" + Dt)
        integrator.addConstrainVelocities()


class RespaPropagator(Propagator):
    """
    This class implements a multiple timescale (MTS) rRESPA propagator :cite:`Tuckerman_1992`
    with `N` force groups, where group 0 goes in the innermost loop (shortest timestep) and group
    `N-1` goes in the outermost loop (largest timestep).

    Parameters
    ----------
        loops : list(int)
            A list of `N` integers, where loops[i] determines how many iterations of force group
            `i` are executed for every iteration of force group `i+1`.

    """
    def __init__(self, loops):
        super(RespaPropagator, self).__init__()
        self.declareVariables()
        self.loops = loops

    def declareVariables(self):
        self.perDofVariables["x0"] = 0
        self.persistent = None

    def addSteps(self, integrator, fraction=1.0):
        integrator.addUpdateContextState()
        self._addSubsteps(integrator, self.loops, fraction)
        integrator.addConstrainVelocities()

    def _addSubsteps(self, integrator, loops, fraction):
        group = len(loops) - 1
        n = loops[group]
        delta_v = "v+Dt*f%d/m" % group
        half = "; Dt=%s*dt" % (0.5*fraction/n)
        full = "; Dt=%s*dt" % (fraction/n)
        for i in range(n):
            integrator.addComputePerDof("v", delta_v + (half if i == 0 else full))
            if group == 0:
                integrator.addComputePerDof("x0", "x")
                integrator.addComputePerDof("x", "x+v*Dt" + full)
                integrator.addConstrainPositions()
                integrator.addComputePerDof("v", "(x-x0)/Dt" + full)
            else:
                self._addSubsteps(integrator, loops[0:group], fraction/n)
            if i == n-1:
                integrator.addComputePerDof("v", delta_v + half)


class BussiThermostatPropagator(Propagator):
    """
    This class implements the Stochastic Velocity Rescaling propagator of Bussi, Donadio, and
    Parrinello :cite:`Bussi_2007`.

    .. math::
        e^{\\delta t \\, iL_\\mathrm{T}} = e^{\\delta t \\alpha \\mathbf{p}^T \\nabla_\\mathbf{p}}

    .. warning::
        This integrator requires non-zero initial velocities for the system particles.

    Parameters
    ----------
        temperature : Number or unit.Quantity
            The temperature of the heat bath (in Kelvin).
        timeConstant : Number or unit.Quantity
            The characteristic time constant of thermostat action (in picoseconds).
        degreesOfFreedom : int
            The number of degrees of freedom in the system, which can be retrieved via function
            :func:`~atomsmm.utils.countDegreesOfFreedom`.

    """
    def __init__(self, temperature, timeConstant, degreesOfFreedom):
        super(BussiThermostatPropagator, self).__init__()
        self.declareVariables()
        self.tau = timeConstant.value_in_unit(unit.picoseconds)
        self.dof = degreesOfFreedom
        kB = unit.BOLTZMANN_CONSTANT_kB*unit.AVOGADRO_CONSTANT_NA
        self.kT = (kB*temperature).value_in_unit(unit.kilojoules_per_mole)

    def declareVariables(self):
        self.globalVariables["R"] = 0
        self.globalVariables["Z"] = 0
        self.globalVariables["ready"] = 0
        self.globalVariables["TwoKE"] = 0
        self.globalVariables["factor"] = 0
        self.persistent = None

    def addSteps(self, integrator, fraction=1.0):
        shape = (self.dof - 2 + self.dof % 2)/2
        d = shape - 1/3
        c = 1/math.sqrt(9*d)
        integrator.addComputeGlobal("ready", "0")
        integrator.beginWhileBlock("ready = 0")
        integrator.addComputeGlobal("R", "gaussian")
        integrator.addComputeGlobal("Z", "1+%s*R" % c)
        integrator.beginWhileBlock("Z <= 0")
        integrator.addComputeGlobal("R", "gaussian")
        integrator.addComputeGlobal("Z", "1+%s*R" % c)
        integrator.endBlock()
        integrator.addComputeGlobal("Z", "Z^3")
        integrator.addComputeGlobal("ready", "step(1-0.0331*y*y-u)")
        integrator.beginIfBlock("ready = 0")
        integrator.addComputeGlobal("ready", "step(0.5*y+%s*(1-Z+log(Z))-log(u))" % d)
        integrator.endBlock()
        integrator.endBlock()
        integrator.addComputeSum("TwoKE", "m*v*v")
        integrator.addComputeGlobal("R", "gaussian")
        expression = "sqrt(A+C*B*(R^2+sumRs)+2*sqrt(C*B*A)*R)"
        expression += "; C = %s/TwoKE" % self.kT
        expression += "; B = 1-A"
        expression += "; A = exp(-dt*%s)" % (fraction/self.tau)
        expression += "; sumRs = 2*%s*Z" % d if self.dof % 2 == 0 else "; sumRs = 2*%s*Z" % d
        integrator.addComputeGlobal("factor", expression)
        integrator.addComputePerDof("v", "factor*v")
