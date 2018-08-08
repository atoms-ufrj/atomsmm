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

from simtk import unit

import atomsmm


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

    .. note::
        One can visualize the steps of a propagator by simply using the `print()` function having
        the propagator object as an argument.

    """
    def __init__(self):
        self.globalVariables = dict()
        self.perDofVariables = dict()
        self.persistent = list()

    def __str__(self):
        return self.integrator().pretty_format()

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
        integrator = atomsmm.integrators.Integrator(stepSize)
        self.addVariables(integrator)
        self.addSteps(integrator)
        return integrator


class ChainedPropagator(Propagator):
    """
    This class combines two propagators :math:`A = e^{\\delta t \\, iL_A}` and
    :math:`B = e^{\\delta t \\, iL_B}` by making :math:`C = A B`, that is,

    .. math::
        e^{\\delta t \\, iL_C} = e^{\\delta t \\, iL_A}
                                 e^{\\delta t \\, iL_B}.

    .. warning::
        Propagators are applied to the system in the right-to-left direction. In general, the effect
        of the chained propagator is non-commutative. Thus, `ChainedPropagator(A,B)` results in a
        time-asymmetric propagators unless `A` and `B` commute.

    .. note::
        It is possible to create nested chained propagators. If, for instance, :math:`B` is
        a chained propagator given by :math:`D E`, then an object instantiated
        by `ChainedPropagator(A,B)` will be a propagator corresponding to
        :math:`A D E`.

    Parameters
    ----------
        A : :class:`Propagator`
            The secondly applied propagator in the chain.
        B : :class:`Propagator`
            The firstly applied propagator in the chain.

    """
    def __init__(self, A, B):
        super(ChainedPropagator, self).__init__()
        self.A = deepcopy(A)
        self.B = deepcopy(B)
        for propagator in [self.A, self.B]:
            self.globalVariables.update(propagator.globalVariables)
            self.perDofVariables.update(propagator.perDofVariables)

    def addSteps(self, integrator, fraction=1.0):
        self.B.addSteps(integrator, fraction)
        self.A.addSteps(integrator, fraction)


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

    Parameters
    ----------
        A : :class:`Propagator`
            The central propagator of a Trotter-Suzuki splitting scheme.
        B : :class:`Propagator`
            The peripheral propagator of a Trotter-Suzuki splitting scheme.

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


class VelocityRescalingPropagator(Propagator):
    """
    This class implements the Stochastic Velocity Rescaling propagator of Bussi, Donadio, and
    Parrinello :cite:`Bussi_2007`, which is a global version of the Langevin thermostat
    :cite:`Bussi_2008`.

    This propagator provides a solution for the following :term:`SDE` :cite:`Bussi_2008`:

    .. math::
        d\\mathbf{p} = \\frac{1}{2\\tau}\\left[\\frac{(N_f-1)k_BT}{2K}-1\\right]\\mathbf{p}dt
                     + \\sqrt{\\frac{k_BT}{2K\\tau}}\\mathbf{p}dW

    The gamma-distributed random numbers required for the solution are generated by using the
    algorithm of Marsaglia and Tsang :cite:`Marsaglia_2000`.

    .. warning::
        An integrator that uses this propagator will fail if no initial velocities are provided to
        the system particles.

    Parameters
    ----------
        temperature : unit.Quantity
            The temperature of the heat bath.
        degreesOfFreedom : int
            The number of degrees of freedom in the system, which can be retrieved via function
            :func:`~atomsmm.utils.countDegreesOfFreedom`.
        timeConstant : unit.Quantity
            The relaxation time of the thermostat.

    """
    def __init__(self, temperature, degreesOfFreedom, timeConstant):
        super(VelocityRescalingPropagator, self).__init__()
        self.declareVariables()
        self.tau = timeConstant.value_in_unit(unit.picoseconds)
        self.dof = degreesOfFreedom
        kB = unit.BOLTZMANN_CONSTANT_kB*unit.AVOGADRO_CONSTANT_NA
        self.kT = (kB*temperature).value_in_unit(unit.kilojoules_per_mole)

    def declareVariables(self):
        self.globalVariables["V"] = 0
        self.globalVariables["X"] = 0
        self.globalVariables["U"] = 0
        self.globalVariables["ready"] = 0
        self.globalVariables["TwoK"] = 0
        self.globalVariables["SumRs"] = 0
        self.persistent = None

    def addSteps(self, integrator, fraction=1.0):
        a = (self.dof - 2 + self.dof % 2)/2
        d = a - 1/3
        c = 1/math.sqrt(9*d)
        integrator.addComputeGlobal("ready", "0")
        integrator.beginWhileBlock("ready < 0.5")
        integrator.addComputeGlobal("X", "gaussian")
        integrator.addComputeGlobal("V", "1+%s*X" % c)
        integrator.beginWhileBlock("V <= 0.0")
        integrator.addComputeGlobal("X", "gaussian")
        integrator.addComputeGlobal("V", "1+%s*X" % c)
        integrator.endBlock()
        integrator.addComputeGlobal("V", "V^3")
        integrator.addComputeGlobal("U", "random")
        integrator.addComputeGlobal("ready", "step(1-0.0331*X^4-U)")
        integrator.beginIfBlock("ready < 0.5")
        integrator.addComputeGlobal("ready", "step(0.5*X^2+%s*(1-V+log(V))-log(U))" % d)
        integrator.endBlock()
        integrator.endBlock()
        integrator.addComputeSum("TwoK", "m*v*v")
        odd = self.dof % 2 == 1
        if odd:
            integrator.addComputeGlobal("X", "gaussian")
        expression = "factor*v"
        expression += "; factor = sqrt(A+C*B*(gaussian^2+sumRs)+2*sqrt(C*B*A)*gaussian)"
        expression += "; C = %s/TwoK" % self.kT
        expression += "; B = 1-A"
        expression += "; A = exp(-dt*%s)" % (fraction/self.tau)
        expression += "; sumRs = %s*V" % (2*d) + ("+X^2" if odd else "")
        # Note: the factor 2 above (multiplying d) is absent in the original paper, but has been
        # added afterwards (see https://sites.google.com/site/giovannibussi/Research/algorithms).
        integrator.addComputePerDof("v", expression)


class NoseHooverLangevinPropagator(Propagator):
    """
    This class implements a Nose-Hoover-Langevin propagator :cite:`Samoletov_2007,Leimkuhler_2009`,
    which is similar to a Nose-Hoover chain :cite:`Tuckerman_1992` of two thermostats, but with the
    second one being a stochastic (Langevin-type) rather than a deterministic thermostat.

    This propagator provides a solution for the following :term:`SDE` system:

    .. math::
        & \\frac{d\\mathbf{p}}{dt} = -\\frac{p_\\eta}{Q} \\mathbf{p} & \\qquad\\mathrm{(S)} \\\\
        & dp_\\eta = (\\mathbf{p}^T\\mathbf{M}^{-1}\\mathbf{p} - N_fk_BT)dt
                   - \\gamma p_\\eta dt + \\sqrt{2\\gamma Qk_BT}dW & \\qquad\\mathrm{(O)}

    As usual, the inertial parameter :math:`Q` is defined as :math:`Q = N_f k_B T \\tau^2`, with
    :math:`\\tau` being a relaxation time :cite:`Tuckerman_1992`. An approximate solution is
    obtained by applying the Trotter-Suzuki splitting formula with the analytical solutions of the
    two equations above taken as if they were independent.

    The solution of Equation "S" is a simple scaling:

    .. math::
        \\mathbf{p}(t) = \\mathbf{p}_0 e^{-\\frac{p_\\eta}{Q}t} \\qquad\\mathrm{(S)}

    Equation "O" represents an Ornstein–Uhlenbeck process, whose solution is:

    .. math::
        p_\\eta(t) = {p_\\eta}_0 e^{-\\gamma t}
                   + \\frac{2K - N_fk_BT}{\\gamma}(1-e^{-\\gamma t})
                   + \\tau k_B T \\sqrt{N_f(1-e^{-2\\gamma t})} R_N, \\qquad\\mathrm{(O)}

    where :math:`K = \\frac{1}{2} \\mathbf{p}^T\\mathbf{M}^{-1}\\mathbf{p}` and :math:`R_N` is a
    normally distributed random number. The splitting solution is, then, given by
    :math:`e^{(\\delta t/2)\\mathcal{L}_S}e^{\\delta t\\mathcal{L}_O}e^{(\\delta t/2)\\mathcal{L}_S}`.

    Parameters
    ----------
        temperature : unit.Quantity
            The temperature of the heat bath.
        degreesOfFreedom : int
            The number of degrees of freedom in the system, which can be retrieved via function
            :func:`~atomsmm.utils.countDegreesOfFreedom`.
        timeConstant : unit.Quantity (time)
            The relaxation time of the Nose-Hoover thermostat.
        frictionCoefficient : unit.Quantity (1/time)
            The friction coefficient of the Langevin thermostat.

    """
    def __init__(self, temperature, degreesOfFreedom, timeConstant, frictionCoefficient):
        super(NoseHooverLangevinPropagator, self).__init__()
        self.declareVariables()
        self.temperature = temperature
        self.degreesOfFreedom = degreesOfFreedom
        self.timeConstant = timeConstant
        self.frictionCoefficient = frictionCoefficient

    def declareVariables(self):
        self.globalVariables["TwoK"] = 0
        self.globalVariables["factor"] = 0
        self.globalVariables["p_NHL"] = 0
        self.persistent = ["p_NHL"]

    def addSteps(self, integrator, fraction=1.0):
        R = unit.BOLTZMANN_CONSTANT_kB*unit.AVOGADRO_CONSTANT_NA
        kT = (R*self.temperature).value_in_unit(unit.kilojoules_per_mole)
        N = self.degreesOfFreedom
        tau = self.timeConstant.value_in_unit(unit.picoseconds)
        gamma = self.frictionCoefficient.value_in_unit(unit.picoseconds**(-1))
        Q = N*kT*tau**2
        integrator.addComputeSum("TwoK", "m*v*v")
        integrator.addComputeGlobal("factor", "exp({}*p_NHL*dt)".format(-0.5*fraction/Q))
        expression = "p_NHL*x+G*(1-x)+{}*sqrt(1-x^2)*gaussian".format(tau*kT*math.sqrt(N))
        expression += "; G = (factor^2*TwoK-{})/{}".format(N*kT, gamma)
        expression += "; x = exp({}*dt)".format(-gamma*fraction)
        integrator.addComputeGlobal("p_NHL", expression)
        integrator.addComputePerDof("v", "factor*exp({}*p_NHL*dt)*v".format(-0.5*fraction/Q))
