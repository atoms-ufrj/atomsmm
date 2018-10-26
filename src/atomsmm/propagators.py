"""
.. module:: propagators
   :platform: Unix, Windows
   :synopsis: a module for defining propagator classes.

.. moduleauthor:: Charlles R. A. Abreu <abreu@eq.ufrj.br>

.. _CustomIntegrator: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.CustomIntegrator.html
.. _VerletIntegrator: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.VerletIntegrator.html

"""

import math

from simtk import unit

import atomsmm
from atomsmm.utils import InputError


class Propagator:
    """
    This is the base class for propagators, which are building blocks for constructing
    CustomIntegrator_ objects in OpenMM. Shortly, a propagator translates the effect of an
    exponential operator like :math:`e^{\\delta t \\, iL}`. This effect can be either the exact
    solution of a system of deterministic or stochastic differential equations or an approximate
    solution obtained by a splitting scheme such as, for instance,
    :math:`e^{\\delta t \\, (iL_A+iL_B)} \\approx e^{\\delta t \\, iL_A} e^{\\delta t \\, iL_B}`.

    .. note::
        One can visualize the steps of a propagator by simply using the `print()` function having
        the propagator object as an argument.

    """
    def __init__(self):
        self.globalVariables = dict()
        self.perDofVariables = dict()
        self.persistent = list()

    kB = unit.BOLTZMANN_CONSTANT_kB*unit.AVOGADRO_CONSTANT_NA

    def __str__(self):
        return self.integrator(1*unit.femtoseconds).pretty_format()

    def addVariables(self, integrator):
        for (name, value) in self.globalVariables.items():
            integrator.addGlobalVariable(name, value)
        for (name, value) in self.perDofVariables.items():
            integrator.addPerDofVariable(name, value)

    def addSteps(self, integrator, fraction=1.0, forceGroup=""):
        pass

    def integrator(self, stepSize):
        """
        This method generates an :class:`Integrator` object which implements the effect of the
        propagator.

        Parameters
        ----------
            stepSize : unit.Quantity
                The step size for integrating the equations of motion.

        Returns
        -------
            :class:`Integrator`

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
        of the chained propagator is non-commutative. Thus, `ChainedPropagator(A, B)` results in a
        time-asymmetric propagator unless `A` and `B` commute.

    .. note::
        It is possible to create nested chained propagators. If, for instance, :math:`B` is
        a chained propagator given by :math:`B = D E`, then an object instantiated
        by `ChainedPropagator(A, B)` will be a propagator corresponding to
        :math:`C = A D E`.

    Parameters
    ----------
        A : :class:`Propagator`
            The secondly applied propagator in the chain.
        B : :class:`Propagator`
            The firstly applied propagator in the chain.

    """
    def __init__(self, A, B):
        super().__init__()
        self.A = A
        self.B = B
        for propagator in [self.A, self.B]:
            self.globalVariables.update(propagator.globalVariables)
            self.perDofVariables.update(propagator.perDofVariables)

    def addSteps(self, integrator, fraction=1.0, forceGroup=""):
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
        a Trotter-Suzuki propagator given by :math:`B = E^{1/2} D E^{1/2}`, then an object
        instantiated by `TrotterSuzukiPropagator(A, B)` will be a propagator corresponding to
        :math:`C = E^{1/4} D^{1/2} E^{1/4} A E^{1/4} D^{1/2} E^{1/4}`.

    Parameters
    ----------
        A : :class:`Propagator`
            The middle propagator of a Trotter-Suzuki splitting scheme.
        B : :class:`Propagator`
            The side propagator of a Trotter-Suzuki splitting scheme.

    """
    def __init__(self, A, B):
        super().__init__()
        self.A = A
        self.B = B
        for propagator in [self.A, self.B]:
            self.globalVariables.update(propagator.globalVariables)
            self.perDofVariables.update(propagator.perDofVariables)

    def addSteps(self, integrator, fraction=1.0, forceGroup=""):
        self.B.addSteps(integrator, 0.5*fraction)
        self.A.addSteps(integrator, fraction)
        self.B.addSteps(integrator, 0.5*fraction)


class SuzukiYoshidaPropagator(Propagator):
    """
    This class splits a propagator :math:`A = e^{\\delta t \\, iL_A}` by using a high-order,
    time-symmetric Suzuki-Yoshida scheme :cite:`Suzuki_1985,Yoshida_1990,Suzuki_1991` given by

    .. math::
        e^{\\delta t \\, iL_A} = \\prod_{i=1}^{n_{sy}} e^{w_i \\delta t \\, iL_A},

    where :math:`n_{sy}` is the number of employed Suzuki-Yoshida weights.

    Parameters
    ----------
        A : :class:`Propagator`
            The propagator to be splitted by the high-order Suzuki-Yoshida scheme.
        nsy : int, optional, default=3
            The number of Suzuki-Yoshida weights to be employed. This must be 3, 7, or 15.

    """
    def __init__(self, A, nsy=3):
        if nsy not in [3, 7, 15]:
            raise InputError("SuzukiYoshidaPropagator accepts nsy = 3, 7, or 15 only")
        super().__init__()
        self.A = A
        self.nsy = nsy
        self.globalVariables.update(self.A.globalVariables)
        self.perDofVariables.update(self.A.perDofVariables)

    def addSteps(self, integrator, fraction=1.0, forceGroup=""):
        if self.nsy == 15:
            weights = [0.102799849391985, -1.96061023297549, 1.93813913762276, -0.158240635368243,
                       -1.44485223686048, 0.253693336566229, 0.914844246229740]
        elif self.nsy == 7:
            weights = [0.784513610477560, 0.235573213359357, -1.17767998417887]
        else:
            weights = [1/(2 - 2**(1/3))]
        for w in list(reversed(weights)) + [1 - 2*sum(weights)] + weights:
            self.A.addSteps(integrator, fraction*w)


class TranslationPropagator(Propagator):
    """
    This class implements a coordinate translation propagator
    :math:`e^{\\delta t \\mathbf{p}^T \\mathbf{M}^{-1} \\nabla_\\mathbf{r}}`.

    Parameters
    ----------
        constrained : bool, optional, default=True
            If `True`, distance constraints are taken into account.

    """
    def __init__(self, constrained=True):
        super().__init__()
        self.constrained = constrained
        if constrained:
            self.perDofVariables["x0"] = 0

    def addSteps(self, integrator, fraction=1.0, forceGroup=""):
        if self.constrained:
            integrator.addComputePerDof("x0", "x")
        integrator.addComputePerDof("x", "x + {}*dt*v".format(fraction))
        if self.constrained:
            integrator.addConstrainPositions()
            integrator.addComputePerDof("v", "(x - x0)/({}*dt)".format(fraction))


class VelocityBoostPropagator(Propagator):
    """
    This class implements a velocity boost propagator
    :math:`e^{\\frac{1}{2} \\delta t \\mathbf{F}^T \\nabla_\\mathbf{p}}`.

    Parameters
    ----------
        constrained : bool, optional, default=True
            If `True`, distance constraints are taken into account.

    """
    def __init__(self, constrained=True):
        super().__init__()
        self.constrained = constrained

    def addSteps(self, integrator, fraction=1.0, forceGroup=""):
        integrator.addComputePerDof("v", "v + {}*dt*f{}/m".format(fraction, forceGroup))
        if self.constrained:
            integrator.addConstrainVelocities()


class MassiveIsokineticPropagator(Propagator):
    """
    This class implements an unconstrained, massive isokinetic propagator. It provides, for every
    degree of freedom in the system, a solution for one of :term:`ODE` systems below.

    1. Force-dependent equations:

    .. math::
        & \\frac{dv}{dt} = \\frac{F}{m} - \\lambda_{F} v \\\\
        & \\frac{dv_1}{dt} = - \\lambda_F v_1 \\\\
        & \\lambda_F = \\frac{F v}{m v^2 + \\frac{1}{2} Q_1 v_1^2}

    where :math:`F` is a constant force. The exact solution for these equations is:

    .. math::
        & v = H \\hat{v} \\\\
        & v_1 = H v_{1,0} \\\\
        & \\text{where:} \\\\
        & \\hat{v} = v_0 \\cosh\\left(\\frac{F t}{\\sqrt{m kT}}\\right) +
                     \\sqrt{\\frac{kT}{m}} \\sinh\\left(\\frac{F t}{\\sqrt{m kT}}\\right) \\\\
        & H = \\sqrt{\\frac{kT}{m \\hat{v}^2 + \\frac{1}{2} Q_1 v_{1,0}^2}} \\\\

    2. Force-indepependent equations:

    .. math::
        & \\frac{dv}{dt} = - \\lambda_{N} v \\\\
        & \\frac{dv_1}{dt} = - (\\lambda_N + v_2) v_1 \\\\
        & \\lambda_N = \\frac{-\\frac{1}{2} Q_1 v_2 v_1^2}{m v^2 + \\frac{1}{2} Q_1 v_1^2}

    where :math:`v_2` is a constant thermostat 'velocity'. In this case, the exact solution is:

    .. math::
        & v = H v_0 \\\\
        & v_1 = H \\hat{v}_1 \\\\
        & \\text{where:} \\\\
        & \\hat{v}_1 = v_{1,0} \\exp(-v_2 t) \\\\
        & H = \\sqrt{\\frac{kT}{m v_0^2 + \\frac{1}{2} Q_1 \\hat{v}_1^2}} \\\\

    Both :term:`ODE` systems above satisfy the massive isokinetic constraint
    :math:`m v^2 + \\frac{1}{2} Q_1 v_1^2 = kT`.

    Parameters
    ----------
        temperature : unit.Quantity
            The temperature to which the configurational sampling should correspond.
        timeScale : unit.Quantity
            A time scale :math:`\\tau` from which to compute the inertial parameter
            :math:`Q_1 = kT\\tau^2`.
        forceDependent : bool
            If `True`, the propagator will solve System 1. If `False`, then System 2 will be solved.

    """
    def __init__(self, temperature, timeScale, forceDependent):
        super().__init__()
        self.globalVariables["kT"] = kT = self.kB*temperature
        self.globalVariables["Q1"] = Q1 = kT*timeScale**2
        self.globalVariables["Q2"] = Q1
        self.perDofVariables["v1"] = 0
        self.perDofVariables["v2"] = 0
        self.persistent = ["kT", "v1", "v2", "Q1", "Q2"]
        self.forceDependent = forceDependent
        self.perDofVariables["H"] = 0

    def addSteps(self, integrator, fraction=1.0, forceGroup=""):
        if self.forceDependent:
            expression = "v*cosh(x) + sqrt(kT/m)*sinh(x)"
            expression += "; x = {}*dt*f{}/sqrt(m*kT)".format(fraction, forceGroup)
            integrator.addComputePerDof("v", expression)
        else:
            integrator.addComputePerDof("v1", "v1*exp(-{}*dt*v2)".format(fraction))
        integrator.addComputePerDof("H", "sqrt(kT/(m*v^2 + 0.5*Q1*v1^2))")
        integrator.addComputePerDof("v", "H*v")
        integrator.addComputePerDof("v1", "H*v1")


class MassiveOrnsteinUhlenbeckPropagator(Propagator):
    """
    This class implements an unconstrained, massive Ornstein-Uhlenbeck (OU) propagator, which
    provides a solution for the following stochastic differential equation for every degree of
    freedom in the system:

    .. math::
        dv = \\frac{G}{m} dt - \\gamma v dt + \\sqrt{\\frac{2 \\gamma kT}{m}} dW.

    There are two options. In the first one, a standard OU process with random and dissipation
    forces only is considered by making :math:`G = 0`. In the second option, a forced OU propagator
    is obtained by making :math:`G = \\frac{Q_1 v_1^2 - kT}{Q_2}`.

    Parameters
    ----------
        temperature : unit.Quantity
            The temperature to which the configurational sampling should correspond.
        timeScale : unit.Quantity, optional, default=None
            A time scale :math:`\\tau` from which to compute the inertial parameters as
            :math:`Q_1 = Q_2 = kT\\tau^2`.
        frictionConstant : unit.Quantity, optional, default=None
            The friction constant :math:`\\gamma` present in the stochastic equation of motion for
            :math:`v_2`.
        forced : bool, optional, default=False
            If True, the propagator carries out an exact solution for the forced OU propagator.

    """
    def __init__(self, temperature, frictionConstant, velocity="v", mass="m", force=None):
        super().__init__()
        self.globalVariables["kT"] = self.kB*temperature
        self.globalVariables["friction"] = frictionConstant
        self.persistent = ["kT", "friction"]
        self.velocity = velocity
        self.mass = mass
        self.force = force

    def addSteps(self, integrator, fraction=1.0, forceGroup=""):
        expression = "x*{} + sqrt(kT*(1 - x^2)/{})*gaussian".format(self.velocity, self.mass)
        if self.force is not None:
            expression += " + G*(1 - x)/({}*friction)".format(self.mass)
            expression += "; G = {}".format(self.force)
        expression += "; x = exp(-{}*dt*friction)".format(fraction)
        integrator.addComputePerDof(self.velocity, expression)


class GenericBoostPropagator(Propagator):
    """
    This class implements a linear boost by providing a solution for the following :term:`ODE` for
    every degree of freedom in the system:

    .. math::
        \\frac{dV}{dt} = \\frac{F}{M}.

    Parameters
    ----------
        velocity : str, optional, default="v"
            The name of a per-dof variable considered as the velocity of each degree of freedom.
        mass : str, optional, default="m"
            The name of a per-dof or global variable considered as the mass of each degree of freedom.
        force : str, optional, default="f"
            The name of a per-dof variable considered as the force acting on each degree of freedom.

    """
    def __init__(self, velocity="v", mass="m", force="f"):
        super().__init__()
        self.velocity = velocity
        self.mass = mass
        self.force = force

    def addSteps(self, integrator, fraction=1.0, forceGroup=""):
        expression = "{} + {}*dt*F/M".format(self.velocity, fraction)
        expression += "; F = {}".format(self.force)
        expression += "; M = {}".format(self.mass)
        integrator.addComputePerDof(self.velocity, expression)


class RespaPropagator(Propagator):
    """
    This class implements a multiple timescale (MTS) rRESPA propagator :cite:`Tuckerman_1992`
    with :math:`N` force groups, where group :math:`0` goes in the innermost loop (shortest
    timestep) and group :math:`N-1` goes in the outermost loop (largest timestep). The complete
    Liouville-like operator corresponding to the equations of motion is split as

    .. math::
        iL = iL_\\mathrm{crust} + iL_\\mathrm{mantle} + iL_\\mathrm{core} +
             iL_\\mathrm{move} + \\sum_{k=0}^{N-1} iL_{\\mathrm{boost}, k}

    In this scheme, :math:`iL_\\mathrm{move}` is the only component that entails changes in the
    atomic coordinates, while :math:`iL_{\\mathrm{boost}, k}` is the only component that depends
    on the forces of group :math:`k`. Therefore, operators :math:`iL_\\mathrm{core}`,
    :math:`iL_\\mathrm{mantle}`, and :math:`iL_\\mathrm{crust}` are reserved to changes in atomic
    velocities due to the action of thermostats, as well as to changes in the thermostat variables
    themselves.

    The rRESPA split is carried out as follows:

    .. math::
        e^{\\Delta t iL} = e^{\\frac{\\Delta t}{2} iL_\\mathrm{crust}}
                           e^{\\Delta t iL_{N-1}}
                           e^{\\frac{\\Delta t}{2} iL_\\mathrm{crust}}

    where

    .. math::
        e^{\\delta t iL_k} = \\begin{cases}
                             \\left(e^{\\frac{\\delta t}{2 N n_k} iL_\\mathrm{mantle}}
                             e^{\\frac{\\delta t}{2 n_k} iL_{\\mathrm{boost}, k}}
                             e^{\\frac{\\delta t}{n_k} iL_{k-1}}
                             e^{\\frac{\\delta t}{2 n_k} iL_{\\mathrm{boost}, k}}
                             e^{\\frac{\\delta t}{2 N n_k} iL_\\mathrm{mantle}}\\right)^{n_k} & k \\geq 0 \\\\
                             e^{\\frac{\\delta t}{2} iL_\\mathrm{move}}
                             e^{\\delta t iL_\\mathrm{core}}
                             e^{\\frac{\\delta t}{2} iL_\\mathrm{move}} & k = -1
                             \\end{cases}

    Parameters
    ----------
        loops : list(int)
            A list of `N` integers, where loops[i] determines how many iterations of force group
            `i` are executed for every iteration of force group `i+1`.
        move : :class:`Propagator`, optional, default=None
            A propagator used to update the coordinate of every atom based on its current velocity.
            If it is `None`, then an unconstrained, linear translation is applied.
        boost : :class:`Propagator`, optional, default=None
            A propagator used to update the velocity of every atom based on the resultant force
            acting on it. If it is `None`, then an unconstrained, linear boosting is applied.
        core : :class:`Propagator`, optional, default=None
            An internal propagator used to control the configurational probability distribution
            sampled by the rRESPA scheme. If it is `None`, then no internal propagator is applied.
        mantle : :class:`Propagator`, optional, default=None
            A middle propagator used to control the configurational probability distribution
            sampled by the rRESPA scheme. If it is `None`, then no middle propagator is applied.
        crust : :class:`Propagator`, optional, default=None
            An external propagator used to control the configurational probability distribution
            sampled by the rRESPA scheme. If it is `None`, then no external propagator is applied.

    """
    def __init__(self, loops, move=None, boost=None, core=None, mantle=None, crust=None):
        super().__init__()
        self.loops = loops
        self.N = len(loops)
        self.move = TranslationPropagator(constrained=False) if move is None else move
        self.boost = VelocityBoostPropagator(constrained=False) if boost is None else boost
        self.core = core
        self.mantle = mantle
        self.crust = crust
        for propagator in [self.move, self.boost, core, mantle, crust]:
            if propagator is not None:
                self.globalVariables.update(propagator.globalVariables)
                self.perDofVariables.update(propagator.perDofVariables)
        for (i, n) in enumerate(self.loops):
            if n > 1:
                self.globalVariables["n{}RESPA".format(i)] = 0
        self.persistent = None

    def addSteps(self, integrator, fraction=1.0, forceGroup=""):
        if self.crust is not None:
            self.crust.addSteps(integrator, 0.5*fraction)
        self._addSubsteps(integrator, len(self.loops)-1, fraction)
        if self.crust is not None:
            self.crust.addSteps(integrator, 0.5*fraction)

    def _addSubsteps(self, integrator, group, fraction):
        if group >= 0:
            n = self.loops[group]
            if n > 1:
                counter = "n{}RESPA".format(group)
                integrator.addComputeGlobal(counter, "0")
                integrator.beginWhileBlock("{} < {}".format(counter, n))
            if self.mantle is not None:
                self.mantle.addSteps(integrator, 0.5*fraction/(n*self.N))
            self.boost.addSteps(integrator, 0.5*fraction/n, str(group))
            self._addSubsteps(integrator, group-1, fraction/n)
            self.boost.addSteps(integrator, 0.5*fraction/n, str(group))
            if self.mantle is not None:
                self.mantle.addSteps(integrator, 0.5*fraction/(n*self.N))
            if n > 1:
                integrator.addComputeGlobal(counter, "{} + 1".format(counter))
                integrator.endBlock()
        elif self.core is None:
            self.move.addSteps(integrator, fraction)
        else:
            self.move.addSteps(integrator, 0.5*fraction)
            self.core.addSteps(integrator, fraction)
            self.move.addSteps(integrator, 0.5*fraction)


class VelocityVerletPropagator(Propagator):
    """
    This class implements a Velocity Verlet propagator with constraints.

    .. math::
        e^{\\delta t \\, iL_\\mathrm{NVE}} = e^{\\frac{1}{2} \\delta t \\mathbf{F}^T \\nabla_\\mathbf{p}}
                                             e^{\\delta t \\mathbf{p}^T \\mathbf{M}^{-1} \\nabla_\\mathbf{r}}
                                             e^{\\frac{1}{2} \\delta t \\mathbf{F}^T \\nabla_\\mathbf{p}}

    .. note::
        In the original OpenMM VerletIntegrator_ class, the implemented propagator is a leap-frog
        version of the Verlet method.

    """
    def __init__(self):
        super().__init__()
        self.declareVariables()

    def declareVariables(self):
        self.perDofVariables["x0"] = 0
        self.persistent = None

    def addSteps(self, integrator, fraction=1.0, forceGroup=""):
        Dt = "; Dt=%s*dt" % fraction
        integrator.addComputePerDof("v", "v+0.5*Dt*f/m" + Dt)
        integrator.addComputePerDof("x0", "x")
        integrator.addComputePerDof("x", "x+Dt*v" + Dt)
        integrator.addConstrainPositions()
        integrator.addComputePerDof("v", "(x-x0)/Dt+0.5*Dt*f/m" + Dt)
        integrator.addConstrainVelocities()


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
        timeScale : unit.Quantity
            The relaxation time of the thermostat.

    """
    def __init__(self, temperature, degreesOfFreedom, timeScale):
        super().__init__()
        self.tau = timeScale.value_in_unit(unit.picoseconds)
        self.dof = degreesOfFreedom
        kB = unit.BOLTZMANN_CONSTANT_kB*unit.AVOGADRO_CONSTANT_NA
        self.kT = (kB*temperature).value_in_unit(unit.kilojoules_per_mole)
        self.declareVariables()

    def declareVariables(self):
        self.globalVariables["V"] = 0
        self.globalVariables["X"] = 0
        self.globalVariables["U"] = 0
        self.globalVariables["ready"] = 0
        self.persistent = None

    def addSteps(self, integrator, fraction=1.0, forceGroup=""):
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
        odd = self.dof % 2 == 1
        if odd:
            integrator.addComputeGlobal("X", "gaussian")
        expression = "vscaling*v"
        expression += "; vscaling = sqrt(A+C*B*(gaussian^2+sumRs)+2*sqrt(C*B*A)*gaussian)"
        expression += "; C = %s/mvv" % self.kT
        expression += "; B = 1-A"
        expression += "; A = exp(-dt*%s)" % (fraction/self.tau)
        expression += "; sumRs = %s*V" % (2*d) + ("+X^2" if odd else "")
        # Note: the vscaling 2 above (multiplying d) is absent in the original paper, but has been
        # added afterwards (see https://sites.google.com/site/giovannibussi/Research/algorithms).
        integrator.addComputePerDof("v", expression)


class NoseHooverPropagator(Propagator):
    """
    This class implements a Nose-Hoove propagator.

    As usual, the inertial parameter :math:`Q` is defined as :math:`Q = N_f k_B T \\tau^2`, with
    :math:`\\tau` being a relaxation time :cite:`Tuckerman_1992`.

    Parameters
    ----------
        temperature : unit.Quantity
            The temperature of the heat bath.
        degreesOfFreedom : int
            The number of degrees of freedom in the system, which can be retrieved via function
            :func:`~atomsmm.utils.countDegreesOfFreedom`.
        timeScale : unit.Quantity (time)
            The relaxation time of the Nose-Hoover thermostat.
        nloops : int, optional, default=1
            Number of RESPA-like subdivisions.

    """
    def __init__(self, temperature, degreesOfFreedom, timeScale, nloops=1):
        super().__init__()
        self.nloops = nloops
        self.globalVariables["LkT"] = degreesOfFreedom*self.kB*temperature
        self.globalVariables["Q"] = degreesOfFreedom*self.kB*temperature*timeScale**2
        self.globalVariables["vscaling"] = 0
        self.globalVariables["p_eta"] = 0
        self.globalVariables["n_NH"] = 0
        self.persistent = ["p_eta"]

    def addSteps(self, integrator, fraction=1.0, forceGroup=""):
        n = self.nloops
        subfrac = fraction/n
        integrator.addComputeGlobal("p_eta", "p_eta + {}*dt*(mvv - LkT)".format(0.5*subfrac))
        integrator.addComputeGlobal("vscaling", "exp(-{}*dt*p_eta/Q)".format(subfrac))
        if n > 2:
            counter = "n_NH"
            integrator.addComputeGlobal(counter, "1")
            integrator.beginWhileBlock("{} < {}".format(counter, n))
            integrator.addComputeGlobal("p_eta", "p_eta + {}*dt*(vscaling^2*mvv - LkT)".format(subfrac))
            integrator.addComputeGlobal("vscaling", "vscaling*exp(-{}*dt*p_eta/Q)".format(subfrac))
            integrator.addComputeGlobal(counter, "{} + 1".format(counter))
            integrator.endBlock()
        integrator.addComputeGlobal("p_eta", "p_eta + {}*dt*(vscaling^2*mvv - LkT)".format(0.5*subfrac))
        integrator.addComputePerDof("v", "vscaling*v")


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
        timeScale : unit.Quantity (time)
            The relaxation time of the Nose-Hoover thermostat.
        frictionConstant : unit.Quantity (1/time)
            The friction coefficient of the Langevin thermostat.

    """
    def __init__(self, temperature, degreesOfFreedom, timeScale, frictionConstant=None):
        super().__init__()
        self.temperature = temperature
        self.degreesOfFreedom = degreesOfFreedom
        self.timeScale = timeScale
        if frictionConstant is None:
            self.frictionConstant = 1/timeScale
        else:
            self.frictionConstant = frictionConstant
        self.declareVariables()

    def declareVariables(self):
        self.globalVariables["vscaling"] = 0
        self.globalVariables["p_NHL"] = 0
        self.persistent = ["p_NHL"]

    def addSteps(self, integrator, fraction=1.0, forceGroup=""):
        R = unit.BOLTZMANN_CONSTANT_kB*unit.AVOGADRO_CONSTANT_NA
        kT = (R*self.temperature).value_in_unit(unit.kilojoules_per_mole)
        N = self.degreesOfFreedom
        tau = self.timeScale.value_in_unit(unit.picoseconds)
        gamma = self.frictionConstant.value_in_unit(unit.picoseconds**(-1))
        Q = N*kT*tau**2
        integrator.addComputeGlobal("vscaling", "exp({}*p_NHL*dt)".format(-0.5*fraction/Q))
        expression = "p_NHL*x+G*(1-x)+{}*sqrt(1-x^2)*gaussian".format(tau*kT*math.sqrt(N))
        expression += "; G = (vscaling^2*mvv-{})/{}".format(N*kT, gamma)
        expression += "; x = exp({}*dt)".format(-gamma*fraction)
        integrator.addComputeGlobal("p_NHL", expression)
        integrator.addComputePerDof("v", "vscaling*exp({}*p_NHL*dt)*v".format(-0.5*fraction/Q))
