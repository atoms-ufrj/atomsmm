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
from atomsmm.utils import kB


class Propagator:
    """
    This is the base class for propagators, which are building blocks for constructing
    CustomIntegrator_ objects in OpenMM. Shortly, a propagator translates the effect of an
    exponential operator like :math:`e^{\\delta t \\, iL}`. This effect can be either the exact
    solution of a system of deterministic or stochastic differential equations or an approximate
    solution obtained by a splitting scheme such as, for instance,
    :math:`e^{\\delta t \\, (iL_A+iL_B)} \\approx e^{\\delta t \\, iL_A} e^{\\delta t \\, iL_B}`.

    """
    def __init__(self):
        self.globalVariables = dict()
        self.perDofVariables = dict()

    def addVariables(self, integrator):
        for (name, value) in self.globalVariables.items():
            integrator.addGlobalVariable(name, value)
        for (name, value) in self.perDofVariables.items():
            integrator.addPerDofVariable(name, value)

    def absorbVariables(self, propagator):
        for (key, value) in propagator.globalVariables.items():
            if key in self.globalVariables and value != self.globalVariables[key]:
                raise InputError('Global variable inconsistency in merged propagators')
        for (key, value) in propagator.perDofVariables.items():
            if key in self.perDofVariables and value != self.perDofVariables[key]:
                raise InputError('Per-dof variable inconsistency in merged propagators')
        self.globalVariables.update(propagator.globalVariables)
        self.perDofVariables.update(propagator.perDofVariables)

    def addSteps(self, integrator, fraction=1.0, force='f'):
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
        integrator = atomsmm.integrators._AtomsMM_Integrator(stepSize)
        self.addVariables(integrator)
        self.addSteps(integrator)
        return integrator


class ChainedPropagator(Propagator):
    """
    This class combines a list of propagators :math:`A1 = e^{\\delta t \\, iL_{A1}}` and
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
    def __init__(self, propagators):
        super().__init__()
        self.propagators = propagators
        for propagator in propagators:
            self.absorbVariables(propagator)

    def addSteps(self, integrator, fraction=1.0, force='f'):
        for propagator in self.propagators:
            propagator.addSteps(integrator, fraction)


class SplitPropagator(Propagator):
    """
    This class splits a propagators :math:`A = e^{\\delta t \\, iL_A}` into a sequence of `n`
    propagators :math:`a = e^{\\frac{\\delta t}{n} \\, iL_A}`, that is,

    .. math::
        e^{\\delta t \\, iL_A} = \\left( e^{\\frac{\\delta t}{n} \\, iL_A} \\right)^n.

    Parameters
    ----------
        A : :class:`Propagator`
            The propagator to be split.
        n : int
            The number of parts.

    """
    def __init__(self, A, n):
        super().__init__()
        self.A = A
        self.absorbVariables(A)
        self.n = n
        self.globalVariables['nSplit'] = 0

    def addSteps(self, integrator, fraction=1.0, force='f'):
        n = self.n
        if n == 1:
            self.A.addSteps(integrator, fraction)
        else:
            integrator.addComputeGlobal('nSplit', '0')
            integrator.beginWhileBlock('nSplit < {}'.format(n))
            self.A.addSteps(integrator, fraction/n)
            integrator.addComputeGlobal('nSplit', 'nSplit + 1')
            integrator.endBlock()


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
            self.absorbVariables(propagator)

    def addSteps(self, integrator, fraction=1.0, force='f'):
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
        super().__init__()
        if nsy not in [1, 3, 7, 15]:
            raise InputError('SuzukiYoshidaPropagator accepts nsy = 1, 3, 7, or 15 only')
        self.A = A
        self.nsy = nsy
        self.absorbVariables(self.A)

    def addSteps(self, integrator, fraction=1.0, force='f'):
        if self.nsy == 15:
            weights = [0.9148442462, 0.2536933366, -1.4448522369, -0.1582406354, 1.9381391376, -1.960610233, 0.1027998494]
        elif self.nsy == 7:
            weights = [0.784513610477560, 0.235573213359357, -1.17767998417887]
        elif self.nsy == 3:
            weights = [1.3512071919596578]
        else:
            weights = []
        for w in weights + [1 - 2*sum(weights)] + list(reversed(weights)):
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
            self.perDofVariables['x0'] = 0

    def addSteps(self, integrator, fraction=1.0, force='f'):
        if self.constrained:
            integrator.addComputePerDof('x0', 'x')
        integrator.addComputePerDof('x', 'x + ({}*dt)*v'.format(fraction))
        if self.constrained:
            integrator.addConstrainPositions()
            integrator.addComputePerDof('v', '(x - x0)/({}*dt)'.format(fraction))


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

    def addSteps(self, integrator, fraction=1.0, force='f'):
        integrator.addComputePerDof('v', 'v + ({}*dt)*{}/m'.format(fraction, force))
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
    def __init__(self, temperature, timeScale, L, forceDependent):
        super().__init__()
        self.globalVariables['Q1'] = kB*temperature*timeScale**2
        self.globalVariables['L'] = L
        self.globalVariables['LkT'] = L*kB*temperature
        self.L = L
        for i in range(L):
            self.perDofVariables['v1_{}'.format(i)] = 1/timeScale
            self.perDofVariables['v2_{}'.format(i)] = 1/timeScale
        self.perDofVariables['H'] = 0
        self.forceDependent = forceDependent

    def addSteps(self, integrator, fraction=1.0, force='f'):
        L = self.L
        v1 = ['v1_{}'.format(i) for i in range(L)]
        v2 = ['v2_{}'.format(i) for i in range(L)]
        if self.forceDependent:
            expression = 'v*cosh(z) + sqrt(LkT/m)*sinh(z)'
            expression += '; z = ({}*dt)*{}/sqrt(m*LkT)'.format(fraction, force)
            integrator.addComputePerDof('v', expression)
        else:
            for i in range(L):
                integrator.addComputePerDof(v1[i], '{}*exp(-({}*dt)*{})'.format(v1[i], fraction, v2[i]))
        sumv1sq = '+'.join(['{}^2'.format(v1[i]) for i in range(L)])
        integrator.addComputePerDof('H', 'sqrt(LkT/(m*v^2 + {}*Q1*({})))'.format(L/(L+1), sumv1sq))
        integrator.addComputePerDof('v', 'H*v')
        for i in range(L):
            integrator.addComputePerDof(v1[i], 'H*{}'.format(v1[i]))


class NewMethodPropagator(Propagator):
    """

    Parameters
    ----------
        temperature : unit.Quantity
            The temperature to which the configurational sampling should correspond.
        timeScale : unit.Quantity
            A time scale :math:`\\tau` from which to compute the inertial parameter
            :math:`Q_1 = kT\\tau^2`.
        L : int
            The parameter L.
        forceDependent : bool
            If `True`, the propagator will solve System 1. If `False`, then System 2 will be solved.

    """
    def __init__(self, temperature, timeScale, L, forceDependent):
        super().__init__()
        self.globalVariables['kT'] = kT = kB*temperature
        self.globalVariables['LkT'] = L*kT
        self.globalVariables['vlim'] = 30.0
        self.perDofVariables['vc'] = math.sqrt(L/(L+1))
        self.perDofVariables['norm'] = 1
        self.forceDependent = forceDependent

    def addSteps(self, integrator, fraction=1.0, force='f'):
        if self.forceDependent:
            expression = 'vs*cosh(fsdt) + sinh(fsdt)'
            expression += '; fsdt = ({}*dt)*{}/(m*vmax)'.format(fraction, force)
        else:
            expression = 'vs*exp(-({}*dt)*v_eta)'.format(fraction)
        expression += '; vs = v/vmax'
        expression += '; vmax=sqrt(LkT/m)'
        expression = 'select(step(vm-vlim),vlim,select(step(vm+vlim),vm,-vlim)); vm={}'.format(expression)
        # expression = 'max(-vlim,min(vm,vlim)); vlim=30; vm={}'.expression
        integrator.addComputePerDof('v', expression)
        integrator.addComputePerDof('norm', 'sqrt(v^2 + vc^2)')
        integrator.addComputePerDof('v', 'sqrt(LkT/m)*v/norm')
        integrator.addComputePerDof('vc', 'vc/norm')


class RestrainedLangevinPropagator(Propagator):
    """

    Parameters
    ----------
        temperature : unit.Quantity
            The temperature to which the configurational sampling should correspond.
        timeScale : unit.Quantity
            A time scale :math:`\\tau` from which to compute the inertial parameter
            :math:`Q_1 = kT\\tau^2`.
        L : int
            The parameter L.
        forceDependent : bool
            If `True`, the propagator will solve System 1. If `False`, then System 2 will be solved.

    """
    def __init__(self, temperature, frictionConstant, L, kind):
        super().__init__()
        self.globalVariables['kT'] = kT = kB*temperature
        self.globalVariables['TwoGammaByL'] = 2*frictionConstant/L
        self.globalVariables['LkT'] = L*kT
        self.globalVariables['friction'] = frictionConstant
        self.globalVariables['vlim'] = 30.0
        self.perDofVariables['vc'] = math.sqrt(L/(L+1))
        self.perDofVariables['norm'] = 1
        self.kind = kind

    def addSteps(self, integrator, fraction=1.0, force='f'):
        expression = 'vs*cosh(fsdt) + sinh(fsdt)'
        if self.kind == 'force':
            expression += '; fsdt = ({}*dt)*{}/(m*vmax)'.format(fraction, force)
        elif self.kind == 'random':
            expression += '; fsdt = sqrt({}*dt*TwoGammaByL)*gaussian'.format(fraction)
        elif self.kind == 'damp':
            expression = 'vs*exp(-({}*dt)*friction)'.format(fraction)
        expression += '; vs = v/vmax'
        expression += '; vmax=sqrt(LkT/m)'
        expression = 'select(step(vm-vlim),vlim,select(step(vm+vlim),vm,-vlim)); vm={}'.format(expression)
        # expression = 'max(-vlim,min(vm,vlim)); vlim=30; vm={}'.expression
        integrator.addComputePerDof('v', expression)
        integrator.addComputePerDof('norm', 'sqrt(v^2 + vc^2)')
        integrator.addComputePerDof('v', 'sqrt(LkT/m)*v/norm')
        integrator.addComputePerDof('vc', 'vc/norm')


class LimitedSpeedLangevinPropagator(Propagator):
    """

    Parameters
    ----------
        temperature : unit.Quantity
            The temperature to which the configurational sampling should correspond.
        timeScale : unit.Quantity
            A time scale :math:`\\tau` from which to compute the inertial parameter
            :math:`Q_1 = kT\\tau^2`.
        L : int
            The parameter L.
        kind : str
            Options are `move`, `boost`, and `bath`.

    """
    def __init__(self, temperature, frictionConstant, L, kind):
        super().__init__()
        self.globalVariables['LkT'] = L*kB*temperature
        self.globalVariables['one'] = 1.0
        self.globalVariables['L'] = L
        # self.globalVariables['plim'] = 250.0  # Use prob distrib to define this values
        self.globalVariables['friction'] = frictionConstant
        self.perDofVariables['p'] = 0.0
        self.perDofVariables['C'] = 0.0
        self.globalVariables['plim'] = 15.0
        self.kind = kind

    def addSteps(self, integrator, fraction=1.0, force='f'):
        if self.kind == 'move':
            integrator.addComputePerDof('x', f'x + sqrt(LkT/m)*tanh(p)*{fraction}*dt')
        elif self.kind == 'boost':
            boost = [
                f' p1 = p + {force}*{fraction}*dt/sqrt(m*LkT)',
                'select(step(p1-plim), plim, select(step(p1+plim), p1, -plim))',
            ]
            integrator.addComputePerDof('p', ';'.join(reversed(boost)))
        elif self.kind == 'bath':
            # integrator.addComputePerDof('C', 'p - x*tanh(p) + 2*sqrt(x/L)*gaussian; x = friction*{}*dt/2'.format(fraction))
            # expressions = [
            #     ' z = friction*{}*dt/2'.format(fraction),
            #     ' v = tanh(p)',
            #     'p - (p + z*v - C)/(one + x*(one - v*v))'
            # ]
            # integrator.addComputePerDof('p', ';'.join(reversed(expressions)))
            n = 1
            expressions = [
                f' alpha = exp(-friction*{fraction/n}*dt/2)',
                ' a = alpha^2',
                ' b = sqrt((one-a^2)/L)',
                ' p1 = p/alpha',
                ' v1 = tanh(p1)',
                ' v2 = alpha*v1',
                ' v3 = v2/sqrt(one - v1^2 + v2^2)',
                ' p2 = 0.5*log((one + v3)/(one - v3))',
                ' p3 = a*p2 + b*gaussian',
                ' v4 = tanh(p3)',
                ' v5 = alpha*v4',
                ' v6 = v5/sqrt(one - v4^2 + v5^2)',
                ' p3 = 0.5*log((one + v6)/(one - v6))',
                'p3/alpha',
            ]
            for i in range(n):
                integrator.addComputePerDof('p', ';'.join(reversed(expressions)))


class LimitedSpeedNHLPropagator(Propagator):
    """

    Parameters
    ----------
        temperature : unit.Quantity
            The temperature to which the configurational sampling should correspond.
        timeScale : unit.Quantity
            A time scale :math:`\\tau` from which to compute the inertial parameter
            :math:`Q_1 = kT\\tau^2`.
        L : int
            The parameter L.
        kind : str
            Options are `move`, `boost`, and `bath`.

    """
    def __init__(self, temperature, timeScale, frictionConstant, L, kind):
        super().__init__()
        self.globalVariables['LkT'] = L*kB*temperature
        self.globalVariables['kT'] = kB*temperature
        self.globalVariables['L'] = L
        self.globalVariables['one'] = 1.0
        self.globalVariables['Q_eta'] = kB*temperature*timeScale**2
        self.globalVariables['friction'] = frictionConstant
        self.perDofVariables['p'] = 0.0
        self.perDofVariables['v_eta'] = 0.0
        self.globalVariables['plim'] = 15.0
        self.kind = kind

    def addSteps(self, integrator, fraction=1.0, force='f'):
        if self.kind == 'move':
            integrator.addComputePerDof('x', 'x + sqrt(LkT/m)*tanh(p)*{}*dt'.format(fraction))
        elif self.kind == 'boost':
            boost = [
                ' p1 = p + {}*{}*dt/sqrt(m*LkT)'.format(force, fraction),
                'select(step(p1-plim), plim, select(step(p1+plim), p1, -plim))',
            ]
            integrator.addComputePerDof('p', ';'.join(reversed(boost)))
        elif self.kind == 'bath':
            kick = 'v_eta + (L*p*tanh(p) - one)*kT*{}*dt/Q_eta'.format(0.5*fraction)
            scaling = 'p*exp(-v_eta*{}*dt)'.format(0.5*fraction)
            stochastic = [
                'a = exp(-friction*{}*dt)'.format(fraction),
                'a*v_eta + sqrt((one - a^2)*kT/Q_eta)*gaussian',
            ]
            integrator.addComputePerDof('v_eta', kick)
            integrator.addComputePerDof('p', scaling)
            integrator.addComputePerDof('v_eta', ';'.join(reversed(stochastic)))
            integrator.addComputePerDof('p', scaling)
            integrator.addComputePerDof('v_eta', kick)


class LimitedSpeedStochasticPropagator(Propagator):
    """

    Parameters
    ----------
        temperature : unit.Quantity
            The temperature to which the configurational sampling should correspond.
        timeScale : unit.Quantity
            A time scale :math:`\\tau` from which to compute the inertial parameter
            :math:`Q_1 = kT\\tau^2`.
        L : int
            The parameter L.
        kind : str
            Options are `move`, `boost`, and `bath`.

    """
    def __init__(self, temperature, timeScale, frictionConstant, L, kind):
        super().__init__()
        kT = kB*temperature
        self.globalVariables['LkT'] = L*kT
        self.globalVariables['kT'] = kT
        self.globalVariables['one'] = 1.0
        self.globalVariables['Lp1'] = L + 1.0
        self.globalVariables['Q_eta'] = kT*timeScale**2
        self.globalVariables['friction'] = frictionConstant
        self.globalVariables['plim'] = 15.0
        self.perDofVariables['p'] = 0.0
        self.perDofVariables['v_eta'] = 0.0
        self.kind = kind

    def addSteps(self, integrator, fraction=1.0, force='f'):
        if self.kind == 'move':
            integrator.addComputePerDof('x', 'x + sqrt(LkT/m)*tanh(p)*{}*dt'.format(fraction))
        elif self.kind == 'boost':
            boost = [
                ' p1 = p + {}*{}*dt/sqrt(m*LkT)'.format(force, fraction),
                'select(step(p1-plim), plim, select(step(p1+plim), p1, -plim))',
            ]
            integrator.addComputePerDof('p', ';'.join(reversed(boost)))
        elif self.kind == 'bath':
            kick = 'v_eta + (Lp1*tanh(p)^2 - one)*kT*{}*dt/Q_eta'.format(0.5*fraction)
            # scaling = [
            #     ' y = -v_eta*{}*dt'.format(0.5*fraction),
            #     ' z = (exp(y+p)-exp(y-p))/2',
            #     'log(z + sqrt(z*z + one))',
            # ]
            scaling = [
                ' v1 = tanh(p)',
                ' v2 = v1*exp(-v_eta*{}*dt)'.format(0.5*fraction),
                ' v3 = v2/sqrt(one - v1^2 + v2^2)',
                '0.5*log((one + v3)/(one - v3))',
            ]
            stochastic = [
                ' a = exp(-friction*{}*dt)'.format(fraction),
                'a*v_eta + sqrt((one - a*a)*kT/Q_eta)*gaussian',
            ]
            integrator.addComputePerDof('v_eta', kick)
            integrator.addComputePerDof('p', ';'.join(reversed(scaling)))
            integrator.addComputePerDof('v_eta', ';'.join(reversed(stochastic)))
            integrator.addComputePerDof('p', ';'.join(reversed(scaling)))
            integrator.addComputePerDof('v_eta', kick)


class LimitedSpeedStochasticVelocityPropagator(Propagator):
    """

    Parameters
    ----------
        temperature : unit.Quantity
            The temperature to which the configurational sampling should correspond.
        timeScale : unit.Quantity
            A time scale :math:`\\tau` from which to compute the inertial parameter
            :math:`Q_1 = kT\\tau^2`.
        L : int
            The parameter L.
        kind : str
            Options are `move`, `boost`, and `bath`.

    """
    def __init__(self, temperature, timeScale, frictionConstant, L, kind):
        super().__init__()
        kT = kB*temperature
        self.globalVariables['LkT'] = L*kT
        self.globalVariables['kT'] = kT
        self.globalVariables['Lfactor'] = (L + 1.0)/L
        self.globalVariables['one'] = 1.0
        self.globalVariables['Q_eta'] = kT*timeScale**2
        self.globalVariables['friction'] = frictionConstant
        self.perDofVariables['vcSq'] = L/(L + 1.0)
        self.perDofVariables['normSq'] = 0.0
        self.perDofVariables['v_eta'] = 0.0
        self.kind = kind

    def update_v(self, integrator, expression):
        integrator.addComputePerDof('v', expression)
        integrator.addComputePerDof('normSq', 'vcSq + v^2')
        integrator.addComputePerDof('v', 'sqrt(LkT/m)*v/sqrt(normSq)')
        integrator.addComputePerDof('vcSq', '(LkT/m)*vcSq/normSq')

    def addSteps(self, integrator, fraction=1.0, force='f'):
        if self.kind == 'move':
            integrator.addComputePerDof('x', 'x + v*{}*dt'.format(fraction))
        elif self.kind == 'boost':
            boost = [
                ' vmax=sqrt(LkT/m)',
                ' z = {}*{}*dt/(m*vmax)'.format(force, fraction),
                'v*cosh(z) + vmax*sinh(z)',
            ]
            self.update_v(integrator, ';'.join(reversed(boost)))
        elif self.kind == 'bath':
            kick = 'v_eta + (Lfactor*m*v*v - kT)*{}*dt/Q_eta'.format(0.5*fraction)
            scaling = 'v*exp(-v_eta*{}*dt)'.format(0.5*fraction)
            stochastic = [
                ' a = exp(-friction*{}*dt)'.format(fraction),
                'a*v_eta + sqrt(kT*(one - a*a)/Q_eta)*gaussian',
            ]
            integrator.addComputePerDof('v_eta', kick)
            self.update_v(integrator, scaling)
            integrator.addComputePerDof('v_eta', ';'.join(reversed(stochastic)))
            self.update_v(integrator, scaling)
            integrator.addComputePerDof('v_eta', kick)


class OrnsteinUhlenbeckPropagator(Propagator):
    """
    This class implements an unconstrained, Ornstein-Uhlenbeck (OU) propagator, which provides a
    solution for the following stochastic differential equation for every degree of freedom in the
    system:

    .. math::
        dV = \\frac{F}{M} dt - \\gamma V dt + \\sqrt{\\frac{2 \\gamma kT}{M}} dW.

    In this equation, `V`, `M`, and `F` are generic forms of velocity, mass, and force. By default,
    the propagator acts on the atomic velocities (`v`) and masses (`m`), while the forces are
    considered as null.

    Parameters
    ----------
        temperature : unit.Quantity
            The temperature to which the configurational sampling should correspond.
        frictionConstant : unit.Quantity, optional, default=None
            The friction constant :math:`\\gamma` present in the stochastic equation.
        velocity : str, optional, default='v'
            The name of a per-dof variable considered as the velocity of each degree of freedom.
        mass : str, optional, default='m'
            The name of a per-dof or global variable considered as the mass associated to each
            degree of freedom.
        force : str, optional, default=None
            The name of a per-dof variable considered as the force acting on each degree of freedom.
            If it is `None`, then this force is considered as null.

    """
    def __init__(self, temperature, frictionConstant, velocity='v', mass='m', force=None,
                 overall=False, **globals):
        super().__init__()
        self.globalVariables['kT'] = kB*temperature
        self.globalVariables['friction'] = frictionConstant
        self.velocity = velocity
        self.mass = mass
        self.force = force
        self.overall = overall
        for key, value in globals.items():
            self.globalVariables[key] = value
        if velocity != 'v':
            if self.overall:
                self.globalVariables[velocity] = 0
            else:
                self.perDofVariables[velocity] = 0

    def addSteps(self, integrator, fraction=1.0, force='f'):
        expression = 'z*{} + sqrt(kT*(1 - z*z)/mass)*gaussian'.format(self.velocity, self.mass)
        if self.force is not None:
            expression += ' + force*(1 - z)/(mass*friction)'
            expression += '; force = {}'.format(self.force)
        expression += '; mass = {}'.format(self.mass)
        expression += '; z = exp(-({}*dt)*friction)'.format(fraction)
        if self.overall:
            integrator.addComputeGlobal(self.velocity, expression)
        else:
            integrator.addComputePerDof(self.velocity, expression)


class GenericBoostPropagator(Propagator):
    """
    This class implements a linear boost by providing a solution for the following :term:`ODE` for
    every degree of freedom in the system:

    .. math::
        \\frac{dV}{dt} = \\frac{F}{M}.

    Parameters
    ----------
        velocity : str, optional, default='v'
            The name of a per-dof variable considered as the velocity of each degree of freedom.
        mass : str, optional, default='m'
            The name of a per-dof or global variable considered as the mass associated to each
            degree of freedom.
        force : str, optional, default='f'
            The name of a per-dof variable considered as the force acting on each degree of freedom.

    Keyword Args
    ------------
        perDof : bool, default=True
            This must be `True` if the propagated velocity is a per-dof variable or `False` if it is
            a global variable.

    """
    def __init__(self, velocity='v', mass='m', force='f', perDof=True, **globals):
        super().__init__()
        self.velocity = velocity
        self.mass = mass
        self.force = force
        for key, value in globals.items():
            self.globalVariables[key] = value
        self.perDof = perDof
        if velocity != 'v':
            if perDof:
                self.perDofVariables[velocity] = 0
            else:
                self.globalVariables[velocity] = 0

    def addSteps(self, integrator, fraction=1.0, force='f'):
        expression = '{} + ({}*dt)*F/M'.format(self.velocity, fraction)
        expression += '; F = {}'.format(self.force)
        expression += '; M = {}'.format(self.mass)
        if self.perDof:
            integrator.addComputePerDof(self.velocity, expression)
        else:
            integrator.addComputeGlobal(self.velocity, expression)


class GenericScalingPropagator(Propagator):
    """
    This class implements scaling by providing a solution for the following :term:`ODE` for
    every degree of freedom in the system:

    .. math::
        \\frac{dV}{dt} = -\\lambda_\\mathrm{damping}*V.

    Parameters
    ----------
        velocity : str
            The name of a per-dof variable considered as the velocity of each degree of freedom.
        damping : str
            The name of a per-dof or global variable considered as the damping parameter associated
            to each degree of freedom.

    """
    def __init__(self, velocity, damping, perDof=True, **globals):
        super().__init__()
        self.velocity = velocity
        self.damping = damping
        self.perDof = perDof
        for key, value in globals.items():
            self.globalVariables[key] = value
        if perDof and velocity != 'v':
            self.perDofVariables[velocity] = 0
        elif not perDof:
            self.globalVariables[velocity] = 0

    def addSteps(self, integrator, fraction=1.0, force='f'):
        expression = '{}*exp(-({}*dt)*{})'.format(self.velocity, fraction, self.damping)
        if self.perDof:
            integrator.addComputePerDof(self.velocity, expression)
        else:
            integrator.addComputeGlobal(self.velocity, expression)


class RespaPropagator(Propagator):
    """
    This class implements a multiple timescale (MTS) rRESPA propagator :cite:`Tuckerman_1992`
    with :math:`N` force groups, where group :math:`0` goes in the innermost loop (shortest
    time step) and group :math:`N-1` goes in the outermost loop (largest time step). The complete
    Liouville-like operator corresponding to the equations of motion is split as

    .. math::
        iL = iL_\\mathrm{move} + \\sum_{k=0}^{N-1} \\left( iL_{\\mathrm{boost}, k} \\right) +
             iL_\\mathrm{core} + \\sum_{k=0}^{N-1} \\left( iL_{\\mathrm{shell}, k} \\right)

    In this scheme, :math:`iL_\\mathrm{move}` is the only component that entails changes in the
    atomic coordinates, while :math:`iL_{\\mathrm{boost}, k}` is the only component that depends
    on the forces of group :math:`k`. Therefore, operator :math:`iL_\\mathrm{core}` and each
    operator :math:`iL_{\\mathrm{shell}, k}` are reserved to changes in atomic velocities due to
    the action of thermostats, as well as to changes in the thermostat variables themselves.

    The rRESPA split can be represented recursively as

    .. math::
        e^{\\Delta t iL} = e^{\\Delta t iL_{N-1}}

    where

    .. math::
        e^{\\delta t iL_k} = \\begin{cases}
                             \\left(e^{\\frac{\\delta t}{2 n_k} iL_{\\mathrm{shell}, k}}
                                    e^{\\frac{\\delta t}{2 n_k} iL_{\\mathrm{boost}, k}}
                                    e^{\\frac{\\delta t}{n_k} iL_{k-1}}
                                    e^{\\frac{\\delta t}{2 n_k} iL_{\\mathrm{boost}, k}}
                                    e^{\\frac{\\delta t}{2 n_k} iL_{\\mathrm{shell}, k}}
                             \\right)^{n_k} & k \\geq 0 \\\\
                             e^{\\frac{\\delta t}{2} iL_\\mathrm{move}}
                             e^{\\delta t iL_\\mathrm{core}}
                             e^{\\frac{\\delta t}{2} iL_\\mathrm{move}} & k = -1
                             \\end{cases}

    Parameters
    ----------
        loops : list(int)
            A list of `N` integers, where `loops[k]` determines how many iterations of force group
            `k` are internally executed for every iteration of force group `k+1`.
        move : :class:`Propagator`, optional, default=None
            A propagator used to update the coordinate of every atom based on its current velocity.
            If it is `None`, then an unconstrained, linear translation is applied.
        boost : :class:`Propagator`, optional, default=None
            A propagator used to update the velocity of every atom based on the resultant force
            acting on it. If it is `None`, then an unconstrained, linear boosting is applied.
        core : :class:`Propagator`, optional, default=None
            An internal propagator to be used for controlling the configurational probability
            distribution sampled by the rRESPA scheme. This propagator will be integrated in the
            innermost loop (shortest time step). If it is `None` (default), then no core propagator
            will be applied.
        shell : dict(int : :class:`Propagator`), optional, default=None
            A dictionary of propagators to be used for controlling the configurational probability
            distribution sampled by the rRESPA scheme. Propagator `shell[k]` will be excecuted in
            both extremities of the loop involving forces of group `k`. If it is `None` (default),
            then no shell propagators will be applied. Dictionary keys must be integers from `0` to
            `N-1` and omitted keys mean that no shell propagators will be considered at those
            particular loop levels.
        has_memory : bool, optional, default=True
            If `True`, integration in the fastest time scale remembers the lattest forces computed
            in all other time scales. To compensate, each remembered force is substracted during the
            integration in its respective time scale. **Warning**: this integration scheme is not
            time-reversal symmetric.

    """
    def __init__(self, loops, move=None, boost=None, core=None, shell=None, **kwargs):
        super().__init__()
        self.loops = loops
        self.N = len(loops)
        self.move = TranslationPropagator(constrained=False) if move is None else move
        self.boost = VelocityBoostPropagator(constrained=False) if boost is None else boost
        self.core = core
        if shell is None:
            self.shell = dict()
        elif set(shell.keys()).issubset(range(self.N)):
            self.shell = shell
        else:
            raise InputError('invalid key(s) in RespaPropagator \'shell\' argument')
        for propagator in [self.move, self.boost, self.core] + list(self.shell.values()):
            if propagator is not None:
                self.absorbVariables(propagator)
        for i, n in enumerate(self.loops):
            if n > 1:
                self.globalVariables['n{}RESPA'.format(i)] = 0

        self.expr = ['f{}'.format(group) for group in range(self.N)]
        for group in range(2, self.N):
            self.expr[group] += '-f{}'.format(group-1)
        self.force = self.expr.copy()

        self._has_memory = kwargs.pop('has_memory', False)
        if self._has_memory:
            for group in range(1, self.N):
                self.perDofVariables['fm{}'.format(group)] = 0.0
                self.force[0] += '+fm{}'.format(group)
                self.force[group] += '-fm{}'.format(group)
        self.force = ['({})'.format(force) for force in self.force]

        self._use_respa_switch = kwargs.pop('use_respa_switch', False)
        self._blitz = kwargs.pop('blitz', False)

    def addSteps(self, integrator, fraction=1.0, force='f'):
        if self._use_respa_switch:
            integrator.addComputeGlobal('respa_switch', '1')
        self._addSubsteps(integrator, self.N-1, fraction)
        if self._use_respa_switch:
            integrator.addComputeGlobal('respa_switch', '0')

    def _internalSplitting(self, integrator, timescale, fraction, shell):
        if self._blitz:
            if self._has_memory and timescale > 0:
                integrator.addComputePerDof('F{}'.format(timescale), 'f{}'.format(timescale))
            else:
                self.boost.addSteps(integrator, fraction, self.force[timescale])
            self._addSubsteps(integrator, timescale-1, fraction)
        else:
            shell and shell.addSteps(integrator, 0.5*fraction)
            if self._has_memory and timescale > 0:
                integrator.addComputePerDof('fm{}'.format(timescale), self.expr[timescale])
            else:
                self.boost.addSteps(integrator, 0.5*fraction, self.force[timescale])
            self._addSubsteps(integrator, timescale-1, fraction)
            self.boost.addSteps(integrator, 0.5*fraction, self.force[timescale])
            shell and shell.addSteps(integrator, 0.5*fraction)

    def _addSubsteps(self, integrator, timescale, fraction):
        if timescale >= 0:
            n = self.loops[timescale]
            if n > 1:
                counter = 'n{}RESPA'.format(timescale)
                integrator.addComputeGlobal(counter, '0')
                integrator.beginWhileBlock('{} < {}'.format(counter, n))
            self._internalSplitting(integrator, timescale, fraction/n, self.shell.get(timescale, None))
            if n > 1:
                integrator.addComputeGlobal(counter, '{} + 1'.format(counter))
                integrator.endBlock()
        elif self.core is None:
            self.move.addSteps(integrator, fraction)
        else:
            self.move.addSteps(integrator, 0.5*fraction)
            self.core.addSteps(integrator, fraction)
            self.move.addSteps(integrator, 0.5*fraction)


class MultipleTimeScalePropagator(RespaPropagator):
    """
    This class implements a Multiple Time-Scale (MTS) propagator using the RESPA method.

    Parameters
    ----------
        loops : list(int)
            A list of `N` integers. Assuming that force group `0` is composed of the fastest forces,
            while group `N-1` is composed of the slowest ones, `loops[k]` determines how many steps
            involving forces of group `k` are internally executed for every step involving those of
            group `k+1`.
        move : :class:`Propagator`, optional, default = None
            A move propagator.
        boost : :class:`Propagator`, optional, default = None
            A boost propagator.
        bath : :class:`Propagator`, optional, default = None
            A bath propagator.

    Keyword Args
    ------------
        scheme : str, optional, default = `middle`
            The splitting scheme used to solve the equations of motion. Available options are
            `middle`, `xi-respa`, `xo-respa`, `side`, and `blitz`.
            If it is `middle` (default), then the bath propagator will be inserted between half-step
            coordinate moves during the fastest-force loops.
            If it is `xi-respa`, `xo-respa`, or `side`, then the bath propagator will be integrated
            in both extremities of each loop concerning one of the `N` time scales, with `xi-respa`
            referring to the time scale of fastest forces (force group `0`), `xo-respa` referring to
            the time scale of the slowest forces (force group `N-1`), and `side` requiring the user
            to select the time scale in which to locate the bath propagator via keyword argument
            `location` (see below).
            If it is `blitz`, then the force-related propagators will be fully integrated at the
            outset of each loop in all time scales and the bath propagator will be integrated
            between half-step coordinate moves during the fastest-force loops.
        location : int, optional, default = None
            The index of the force group (from `0` to `N-1`) that defines the time scale in which
            the bath propagator will be located. This is only meaningful if keyword `scheme` is set
            to `side` (see above).
        nsy : int, optional, default = 1
            The number of Suzuki-Yoshida terms to factorize the bath propagator. Valid options are
            1, 3, 7, and 15.
        nres : int, optional, default = 1
            The number of RESPA-like subdivisions to factorize the bath propagator.

    .. warning::
        The `xo-respa` and `xi-respa` schemes implemented here are slightly different from the ones
        described in the paper by Leimkuhler, Margul, and Tuckerman :cite:`Leimkuhler_2013`.

    """
    def __init__(self, loops, move=None, boost=None, bath=None, **kwargs):
        scheme = kwargs.pop('scheme', 'middle')
        location = kwargs.pop('location', 0)
        nres = kwargs.pop('nres', 1)
        nsy = kwargs.pop('nsy', 1)
        if nres > 1:
            bath = SplitPropagator(bath, nres)
        if nsy > 1:
            bath = SuzukiYoshidaPropagator(bath, nsy)
        if scheme == 'middle':
            super().__init__(loops, move=move, boost=boost, core=bath, **kwargs)
        elif scheme == 'blitz':
            super().__init__(loops, move=move, boost=boost, core=bath, blitz=True, **kwargs)
        elif scheme in ['xi-respa', 'xo-respa', 'side']:
            level = location if scheme == 'side' else (0 if scheme == 'xi-respa' else len(loops)-1)
            super().__init__(loops, move=move, boost=boost, shell={level: bath}, **kwargs)
        else:
            raise InputError('wrong value of scheme parameter')


class SIN_R_Propagator(MultipleTimeScalePropagator):
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
        loops : list(int)
            See description in :class:`MultipleTimeScaleIntegrator`.
        temperature : unit.Quantity
            The temperature to which the configurational sampling should correspond.
        timeScale : unit.Quantity
            A time scale :math:`\\tau` from which the inertial parameters are computed as
            :math:`Q_1 = Q_2 = kT\\tau^2`.
        frictionConstant : unit.Quantity
            The friction constant :math:`\\gamma` present in the stochastic equation of motion for
            per-DOF thermostat variable :math:`v_2`.
        **kwargs : keyword arguments
            The same keyword arguments of class :class:`MultipleTimeScalePropagator` apply here.

    """
    def __init__(self, loops, temperature, timeScale, frictionConstant, **kwargs):
        L = kwargs.pop('L', 1)
        split = kwargs.pop('split', False)
        isoF = MassiveIsokineticPropagator(temperature, timeScale, L, forceDependent=True)
        isoN = MassiveIsokineticPropagator(temperature, timeScale, L, forceDependent=False)
        v1 = ['v1_{}'.format(i) for i in range(L)]
        v2 = ['v2_{}'.format(i) for i in range(L)]
        Q2 = kB*temperature*timeScale**2
        OU = []
        boost = []
        for i in range(L):
            OU.append(OrnsteinUhlenbeckPropagator(temperature, frictionConstant, v2[i], 'Q2',
                                                  None if split else f'Q1*{v1[i]}^2 - kT', Q2=Q2))
            if split:
                boost.append(GenericBoostPropagator(v2[i], 'Q2', f'Q1*{v1[i]}^2 - kT', Q2=Q2))
        DOU = ChainedPropagator(OU)
        if split:
            boost = ChainedPropagator(boost)
            DOU = TrotterSuzukiPropagator(DOU, boost)
        bath = TrotterSuzukiPropagator(DOU, isoN)
        super().__init__(loops, None, isoF, bath, **kwargs)


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
        self.perDofVariables['x0'] = 0

    def addSteps(self, integrator, fraction=1.0, force='f'):
        Dt = '; Dt=%s*dt' % fraction
        integrator.addComputePerDof('v', 'v+0.5*Dt*f/m' + Dt)
        integrator.addComputePerDof('x0', 'x')
        integrator.addComputePerDof('x', 'x+Dt*v' + Dt)
        integrator.addConstrainPositions()
        integrator.addComputePerDof('v', '(x-x0)/Dt+0.5*Dt*f/m' + Dt)
        integrator.addConstrainVelocities()


class UnconstrainedVelocityVerletPropagator(Propagator):
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
    def addSteps(self, integrator, fraction=1.0, force='f'):
        integrator.addComputePerDof('v', f'v+0.5*{fraction}*dt*f/m')
        integrator.addComputePerDof('x', f'x+{fraction}*dt*v')
        integrator.addComputePerDof('v', f'v+0.5*{fraction}*dt*f/m')


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
        self.globalVariables['V'] = 0
        self.globalVariables['X'] = 0
        self.globalVariables['U'] = 0
        self.globalVariables['ready'] = 0

    def addSteps(self, integrator, fraction=1.0, force='f'):
        a = (self.dof - 2 + self.dof % 2)/2
        d = a - 1/3
        c = 1/math.sqrt(9*d)
        integrator.addComputeGlobal('ready', '0')
        integrator.beginWhileBlock('ready < 0.5')
        integrator.addComputeGlobal('X', 'gaussian')
        integrator.addComputeGlobal('V', '1+%s*X' % c)
        integrator.beginWhileBlock('V <= 0.0')
        integrator.addComputeGlobal('X', 'gaussian')
        integrator.addComputeGlobal('V', '1+%s*X' % c)
        integrator.endBlock()
        integrator.addComputeGlobal('V', 'V^3')
        integrator.addComputeGlobal('U', 'random')
        integrator.addComputeGlobal('ready', 'step(1-0.0331*X^4-U)')
        integrator.beginIfBlock('ready < 0.5')
        integrator.addComputeGlobal('ready', 'step(0.5*X^2+%s*(1-V+log(V))-log(U))' % d)
        integrator.endBlock()
        integrator.endBlock()
        odd = self.dof % 2 == 1
        if odd:
            integrator.addComputeGlobal('X', 'gaussian')
        expression = 'vscaling*v'
        expression += '; vscaling = sqrt(A+C*B*(gaussian^2+sumRs)+2*sqrt(C*B*A)*gaussian)'
        expression += '; C = %s/mvv' % self.kT
        expression += '; B = 1-A'
        expression += '; A = exp(-dt*%s)' % (fraction/self.tau)
        expression += '; sumRs = %s*V' % (2*d) + ('+X^2' if odd else '')
        # Note: the vscaling 2 above (multiplying d) is absent in the original paper, but has been
        # added afterwards (see https://sites.google.com/site/giovannibussi/Research/algorithms).
        integrator.addComputePerDof('v', expression)


class NoseHooverPropagator(Propagator):
    """
    This class implements a Nose-Hoover propagator.

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
        self.globalVariables['LkT'] = degreesOfFreedom*kB*temperature
        self.globalVariables['Q'] = degreesOfFreedom*kB*temperature*timeScale**2
        self.globalVariables['vscaling'] = 0
        self.globalVariables['p_eta'] = 0
        self.globalVariables['n_NH'] = 0

    def addSteps(self, integrator, fraction=1.0, force='f'):
        n = self.nloops
        subfrac = fraction/n
        integrator.addComputeGlobal('p_eta', 'p_eta + ({}*dt)*(mvv - LkT)'.format(0.5*subfrac))
        integrator.addComputeGlobal('vscaling', 'exp(-({}*dt)*p_eta/Q)'.format(subfrac))
        if n > 2:
            counter = 'n_NH'
            integrator.addComputeGlobal(counter, '1')
            integrator.beginWhileBlock('{} < {}'.format(counter, n))
            integrator.addComputeGlobal('p_eta', 'p_eta + ({}*dt)*(vscaling^2*mvv - LkT)'.format(subfrac))
            integrator.addComputeGlobal('vscaling', 'vscaling*exp(-({}*dt)*p_eta/Q)'.format(subfrac))
            integrator.addComputeGlobal(counter, '{} + 1'.format(counter))
            integrator.endBlock()
        integrator.addComputeGlobal('p_eta', 'p_eta + ({}*dt)*(vscaling^2*mvv - LkT)'.format(0.5*subfrac))
        integrator.addComputePerDof('v', 'vscaling*v')


class MassiveNoseHooverPropagator(Propagator):
    """
    This class implements a massive Nose-Hoover propagator.

    As usual, the inertial parameter :math:`Q` is defined as :math:`Q = N_f k_B T \\tau^2`, with
    :math:`\\tau` being a relaxation time :cite:`Tuckerman_1992`.

    Parameters
    ----------
        temperature : unit.Quantity
            The temperature of the heat bath.
        timeScale : unit.Quantity (time)
            The relaxation time of the Nose-Hoover thermostat.
        nloops : int, optional, default=1
            Number of RESPA-like subdivisions.

    """
    def __init__(self, temperature, timeScale, nloops=1):
        super().__init__()
        self.nloops = nloops
        self.globalVariables['kT'] = kB*temperature
        self.globalVariables['Q'] = kB*temperature*timeScale**2
        self.globalVariables['nMNH'] = 0
        self.perDofVariables['p_eta'] = 0

    def addSteps(self, integrator, fraction=1.0, force='f'):
        subfrac = fraction/self.nloops
        if self.nloops > 1:
            integrator.addComputeGlobal('nMNH', '0')
            integrator.beginWhileBlock(f'nMNH < {self.nloops}')
        integrator.addComputePerDof('p_eta', f'p_eta + ({0.5*subfrac}*dt)*(m*v^2 - kT)')
        integrator.addComputePerDof('v', f'v*exp(-({subfrac}*dt)*p_eta/Q)')
        integrator.addComputePerDof('p_eta', f'p_eta + ({0.5*subfrac}*dt)*(m*v^2 - kT)')
        if self.nloops > 1:
            integrator.addComputeGlobal('nMNH', 'nMNH + 1')
            integrator.endBlock()


class MassiveGeneralizedGaussianMomentPropagator(Propagator):
    """
    This class implements a massive Generalized Gaussian Moment propagator.

    As usual, the inertial parameter :math:`Q` is defined as :math:`Q = N_f k_B T \\tau^2`, with
    :math:`\\tau` being a relaxation time :cite:`Tuckerman_1992`.

    Parameters
    ----------
        temperature : unit.Quantity
            The temperature of the heat bath.
        timeScale : unit.Quantity (time)
            The relaxation time of the Nose-Hoover thermostat.
        nloops : int, optional, default=1
            Number of RESPA-like subdivisions.

    """
    def __init__(self, temperature, timeScale, nloops=1):
        super().__init__()
        self.nloops = nloops
        self.globalVariables['kT'] = kB*temperature
        self.globalVariables['Q1'] = kB*temperature*timeScale**2
        self.globalVariables['Q2'] = 2*(kB*temperature)**3*timeScale**2
        self.globalVariables['nGGM'] = 0
        self.perDofVariables['p1'] = 0
        self.perDofVariables['p2'] = 0

    def addSteps(self, integrator, fraction=1.0, force='f'):
        subfrac = fraction/self.nloops
        if self.nloops > 1:
            integrator.addComputeGlobal('nGGM', '0')
            integrator.beginWhileBlock(f'nGGM < {self.nloops}')
        integrator.addComputePerDof('p1', f'p1 + ({subfrac/2}*dt)*(m*v^2 - kT)')
        integrator.addComputePerDof('p2', f'p2 + ({subfrac/2}*dt)*(m^2*v^4/3 - kT^2)')
        expressions = [
            f'v1 = v*exp(-{subfrac/2}*dt*(p1/Q1 + kT*p2/Q2))',
            'alpha = p2/(3*m*Q2)',
            f'v2 = v1/sqrt(1 + 2*v1^2*alpha*{subfrac}*dt)',
            f'v2*exp(-{subfrac/2}*dt*(p1/Q1 + kT*p2/Q2))',
        ]
        integrator.addComputePerDof('v', ';'.join(reversed(expressions)))
        integrator.addComputePerDof('p2', f'p2 + ({subfrac/2}*dt)*(m^2*v^4/3 - kT^2)')
        integrator.addComputePerDof('p1', f'p1 + ({subfrac/2}*dt)*(m*v^2 - kT)')
        if self.nloops > 1:
            integrator.addComputeGlobal('nGGM', 'nGGM + 1')
            integrator.endBlock()


class NoseHooverChainPropagator(Propagator):
    """
    This class implements a Nose-Hoover chain :cite:`Tuckerman_1992` with two global thermostats.

    This propagator provides a solution for the following :term:`ODE` system:

    .. math::
        & \\frac{d\\mathbf{p}}{dt} = -\\frac{p_{\\eta,1}}{Q_1} \\mathbf{p} \\\\
        & \\frac{dp_{\\eta,1}}{dt} = \\mathbf{p}^T\\mathbf{M}^{-1}\\mathbf{p} - N_f k_B T
                   - \\frac{p_{\\eta,2}}{Q_2} p_{\\eta,1} \\\\
        & \\frac{dp_{\\eta,2}}{dt} = \\frac{p_{\\eta,1}^2}{Q_1} - k_B T

    As usual, the inertial parameter :math:`Q` is defined as :math:`Q = N_f k_B T \\tau^2`, with
    :math:`\\tau` being a relaxation time :cite:`Tuckerman_1992`. An approximate solution is
    obtained by applying the Trotter-Suzuki splitting formula:

    .. math::
        e^{(\\delta t/2)\\mathcal{L}_{B2}}
        e^{(\\delta t/2)\\mathcal{L}_{S1}}
        e^{(\\delta t/2)\\mathcal{L}_{B1}}
        e^{(\\delta t)\\mathcal{L}_{S}}
        e^{(\\delta t/2)\\mathcal{L}_{B1}}
        e^{(\\delta t/2)\\mathcal{L}_{S1}}
        e^{(\\delta t/2)\\mathcal{L}_{B2}}

    Each exponential operator above is the solution of a differential equation.

    Equation 'B2' is a boost of thermostat 2, whose solution is:

    .. math::
        p_{\\eta,2}(t) = p_{\\eta,2}^0 +\\left(\\frac{p_{\\eta,1}^2}{Q_1} - k_B T\\right) t

    Equation 'S1' is a scaling of thermostat 1, whose solution is:

    .. math::
        p_{\\eta,1}(t) = p_{\\eta,1}^0 e^{-\\frac{p_{\\eta,2}}{Q_2} t}

    Equation 'B1' is a boost of thermostat 1, whose solution is:

    .. math::
        p_{\\eta,1}(t) = p_{\\eta,1}^0 + \\left(\\mathbf{p}^T\\mathbf{M}^{-1}\\mathbf{p} - N_f k_B T\\right) t

    Equation 'S' is a scaling of particle momenta, whose solution is:

    .. math::
        \\mathbf{p}(t) = \\mathbf{p}_0 e^{-\\frac{p_{\\eta,1}}{Q_1} t}

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
        self.globalVariables['vscaling'] = 0
        self.globalVariables['p_NHC_1'] = 0
        self.globalVariables['p_NHC_2'] = 0

    def addSteps(self, integrator, fraction=1.0, force='f'):
        R = unit.BOLTZMANN_CONSTANT_kB*unit.AVOGADRO_CONSTANT_NA
        kT = (R*self.temperature).value_in_unit(unit.kilojoules_per_mole)
        NkT = self.degreesOfFreedom*kT
        tau = self.timeScale.value_in_unit(unit.picoseconds)
        Q1 = NkT*tau**2
        Q2 = kT*tau**2
        integrator.addComputeGlobal('p_NHC_2', f'p_NHC_2 + (p_NHC_1^2/{Q1}-{kT})*{0.5*fraction}*dt')
        integrator.addComputeGlobal('p_NHC_1', f'p_NHC_1*exp(-{0.5*fraction/Q2}*p_NHC_2*dt)')
        integrator.addComputeGlobal('p_NHC_1', f'p_NHC_1 + (mvv-{NkT})*{0.5*fraction}*dt')
        integrator.addComputeGlobal('vscaling', f'exp(-{fraction/Q1}*p_NHC_1*dt)')
        integrator.addComputeGlobal('p_NHC_1', f'p_NHC_1 + (vscaling^2*mvv-{NkT})*{0.5*fraction}*dt')
        integrator.addComputeGlobal('p_NHC_1', f'p_NHC_1*exp(-{0.5*fraction/Q2}*p_NHC_2*dt)')
        integrator.addComputeGlobal('p_NHC_2', f'p_NHC_2 + (p_NHC_1^2/{Q1}-{kT})*{0.5*fraction}*dt')
        integrator.addComputePerDof('v', 'vscaling*v')


class NoseHooverLangevinPropagator(Propagator):
    """
    This class implements a Nose-Hoover-Langevin propagator :cite:`Samoletov_2007,Leimkuhler_2009`,
    which is similar to a Nose-Hoover chain :cite:`Tuckerman_1992` of two thermostats, but with the
    second one being a stochastic (Langevin-type) rather than a deterministic thermostat.

    This propagator provides a solution for the following :term:`SDE` system:

    .. math::
        & d\\mathbf{p} = -\\frac{p_\\eta}{Q} \\mathbf{p} dt & \\qquad\\mathrm{(S)} \\\\
        & dp_\\eta = (\\mathbf{p}^T\\mathbf{M}^{-1}\\mathbf{p} - N_f k_B T)dt
                   - \\gamma p_\\eta dt + \\sqrt{2\\gamma Q k_B T}dW & \\qquad\\mathrm{(O)}

    As usual, the inertial parameter :math:`Q` is defined as :math:`Q = N_f k_B T \\tau^2`, with
    :math:`\\tau` being a relaxation time :cite:`Tuckerman_1992`. An approximate solution is
    obtained by applying the Trotter-Suzuki splitting formula:

    .. math::
        e^{(\\delta t/2)\\mathcal{L}_B}
        e^{(\\delta t/2)\\mathcal{L}_S}
        e^{\\delta t\\mathcal{L}_O}
        e^{(\\delta t/2)\\mathcal{L}_S}
        e^{(\\delta t/2)\\mathcal{L}_B}

    Each exponential operator above is the solution of a differential equation.

    Equation 'B' is a boost, whose solution is:

    .. math::
        p_\\eta(t) = {p_\\eta}_0 + (\\mathbf{p}^T\\mathbf{M}^{-1}\\mathbf{p} - N_f k_B T) t

    Equation 'S' is a scaling, whose solution is:

    .. math::
        \\mathbf{p}(t) = \\mathbf{p}_0 e^{-\\frac{p_\\eta}{Q} t}

    Equation 'O' is an Ornstein–Uhlenbeck process, whose solution is:

    .. math::
        p_\\eta(t) = {p_\\eta}_0 e^{-\\gamma t}
                   + \\sqrt{\\frac{k_B T}{Q}(1-e^{-2\\gamma t})} R_N

    where :math:`R_N` is a normally distributed random number.

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
        self.globalVariables['vscaling'] = 0
        self.globalVariables['p_NHL'] = 0

    def addSteps(self, integrator, fraction=1.0, force='f'):
        R = unit.BOLTZMANN_CONSTANT_kB*unit.AVOGADRO_CONSTANT_NA
        kT = (R*self.temperature).value_in_unit(unit.kilojoules_per_mole)
        NkT = self.degreesOfFreedom*kT
        tau = self.timeScale.value_in_unit(unit.picoseconds)
        Q = NkT*tau**2
        gamma = self.frictionConstant.value_in_unit(unit.picoseconds**(-1))
        integrator.addComputeGlobal('p_NHL', f'p_NHL + (mvv-{NkT})*{0.5*fraction}*dt')
        integrator.addComputeGlobal('vscaling', f'exp(-{0.5*fraction/Q}*p_NHL*dt)')
        expression = f'p_NHL*x + sqrt({kT/Q}*(1-x^2))*gaussian; x = exp(-{gamma*fraction}*dt)'
        integrator.addComputeGlobal('p_NHL', expression)
        integrator.addComputeGlobal('vscaling', f'vscaling*exp(-{0.5*fraction/Q}*p_NHL*dt)')
        integrator.addComputeGlobal('p_NHL', f'p_NHL + (vscaling^2*mvv-{NkT})*{0.5*fraction}*dt')
        integrator.addComputePerDof('v', 'vscaling*v')


class RegulatedTranslationPropagator(Propagator):
    """
    An unconstrained, regulated translation propagator which provides, for every degree of freedom
    in the system, a solution for the following :term:`ODE`:

    .. math::
        \\frac{dr_i}{dt} = c_i \\tanh\\left(\\frac{\\alpha_n p_i}{m_i c_i}\\right)

    where :math:`c_i = \\sqrt{\\frac{\\alpha_n n k_B T}{m_i}}` is the speed limit for such degree
    of freedom and, by default, :math:`\\alpha_n = \\frac{n+1}{n}`.
    The exact solution for this equation is:

    .. math::
        r_i(t) = r_i^0 + c_i \\mathrm{tanh}\\left(\\frac{\\alpha_n p}{m c_i}\\right) t

    where :math:`r_i^0` is the initial coordinate.

    Parameters
    ----------
        temperature : unit.Quantity
            The temperature to which the configurational sampling should correspond.
        n : int or float
            The regulating parameter.

    Keyword args
    ------------
        alpha_n : int or float, default=1
            Another regulating parameter.

    """
    def __init__(self, temperature, n, alpha_n=1):
        super().__init__()
        self._alpha = alpha_n
        self.globalVariables['nakT'] = self._alpha*n*kB*temperature

    def addSteps(self, integrator, fraction=1.0, force='f'):
        alpha = self._alpha
        integrator.setKineticEnergyExpression(f'0.5*m*(c*tanh({alpha}*v/c))^2; c=sqrt(ankT/m)')
        integrator.addComputePerDof('x', f'x + c*tanh({alpha}*v/c)*{fraction}*dt; c=sqrt(nakT/m)')


class RegulatedBoostPropagator(Propagator):
    """
    An unconstrained, regulated boost propagator which provides, for every degree of freedom in the
    system, a solution for the following :term:`ODE`:

    .. math::
        \\frac{dp_i}{dt} = F_i

    where :math:`F_i` is a constant force. The exact solution for this equation is:

    .. math::
        p_i(t) = p_i^0 + F_i t

    where :math:`p_i^0` is the initial momentum.

    """
    def addSteps(self, integrator, fraction=1.0, force='f'):
        integrator.addComputePerDof('v', f'v + {force}*{fraction}*dt/m')


class RegulatedMassiveNoseHooverLangevinPropagator(Propagator):
    """
    This class implements a regulated version of the massive Nose-Hoover-Langevin propagator
    :cite:`Samoletov_2007,Leimkuhler_2009`. It provides, for every degree of freedom in the system,
    a solution for the following :term:`SDE` system:

    .. math::
        & dp_i = -v_{\\eta_i} p_i dt \\\\
        & dv_{\\eta_i} = \\frac{p_i v_i - k_B T}{Q} dt
                   - \\gamma v_{\\eta_i} dt + \\sqrt{\\frac{2\\gamma k_B T}{Q}}dW_i,

    where:

    .. math::
        v_i = c_i \\tanh\\left(\\frac{\\alpha_n p_i}{m_i c_i}\\right).

    Here, :math:`c_i = \\sqrt{\\frac{\\alpha_n n k_B T}{m_i}}` is the speed limit for such degree
    of freedom and, by default, :math:`\\alpha_n = \\frac{n+1}{n}`.
    As usual, the inertial parameter :math:`Q` is defined as :math:`Q = k_B T \\tau^2`, with
    :math:`\\tau` being a relaxation time :cite:`Tuckerman_1992`. An approximate solution is
    obtained by applying the Trotter-Suzuki splitting formula:

    .. math::
        e^{\\delta t\\mathcal{L}} =
        e^{(\\delta t/2)\\mathcal{L}_B}
        e^{(\\delta t/2)\\mathcal{L}_S}
        e^{\\delta t\\mathcal{L}_O}
        e^{(\\delta t/2)\\mathcal{L}_S}
        e^{(\\delta t/2)\\mathcal{L}_B}

    Each exponential operator above is the solution of a differential equation.

    Equation 'B' is a boost, whose solution is:

    .. math::
        v_{\\eta_i}(t) = v_{\\eta_i}^0 + \\frac{p_i v_i - k_B T}{Q} t

    Equation 'S' is a scaling, whose solution is:

    .. math::
        p_i(t) = p_i^0 e^{-v_{\\eta_i} t}

    Equation 'O' is an Ornstein–Uhlenbeck process, whose solution is:

    .. math::
        v_{\\eta_i}(t) = v_{\\eta_i}^0 e^{-\\gamma t}
                   + \\sqrt{\\frac{k_B T}{Q}(1-e^{-2\\gamma t})} R_{N,i}

    where :math:`R_{N,i}` is a normally distributed random number.

    Parameters
    ----------
        temperature : unit.Quantity
            The temperature of the heat bath.
        n : int or float
            The regulating parameter.
        timeScale : unit.Quantity (time)
            The relaxation time of the Nose-Hoover thermostat.
        frictionConstant : unit.Quantity (1/time)
            The friction coefficient of the Langevin thermostat.

    Keyword args
    ------------
        alpha_n : int or float, default=1
            Another regulating parameter.

    """
    def __init__(self, temperature, n, timeScale, frictionConstant, alpha_n=1, split=False):
        super().__init__()
        self._alpha = alpha_n
        self._split = split
        kT = kB*temperature
        Q = kT*timeScale**2
        self.globalVariables['kT'] = kT
        self.globalVariables['ankT'] = self._alpha*n*kT
        self.globalVariables['Q'] = Q
        self.globalVariables['omega'] = 1/timeScale
        self.globalVariables['friction'] = frictionConstant
        self.perDofVariables['v_eta'] = 0

    def addSteps(self, integrator, fraction=1.0, force='f'):
        alpha = self._alpha
        G_definition = f'; G=(m*v*c*tanh({alpha}*v/c) - kT)/Q'
        G_definition += '; c=sqrt(ankT/m)'
        boost = f'v_eta + G*{0.5*fraction}*dt' + G_definition
        scaling = f'v*exp(-v_eta*{0.5*fraction}*dt)'
        if self._split:
            OU = 'v_eta*z + omega*sqrt(1-z^2)*gaussian'
        else:
            OU = 'v_eta*z + G*(1-z)/friction + omega*sqrt(1-z^2)*gaussian' + G_definition
        OU += f'; z=exp(-friction*{fraction}*dt)'
        self._split and integrator.addComputePerDof('v_eta', boost)
        integrator.addComputePerDof('v', scaling)
        integrator.addComputePerDof('v_eta', OU)
        integrator.addComputePerDof('v', scaling)
        self._split and integrator.addComputePerDof('v_eta', boost)


class TwiceRegulatedMassiveNoseHooverLangevinPropagator(Propagator):
    """
    This class implements a doubly-regulated version of the massive Nose-Hoover-Langevin propagator
    :cite:`Samoletov_2007,Leimkuhler_2009`. It provides, for every degree of freedom in the system,
    a solution for the following :term:`SDE` system:

    .. math::
        & dp_i = -v_{\\eta_i} m_i v_i dt \\\\
        & dv_{\\eta_i} = \\frac{1}{Q}\\left(\\frac{n+1}{n \\alpha_n} m_i v_i^2 - k_B T\\right) dt
                - \\gamma v_{\\eta_i} dt + \\sqrt{\\frac{2\\gamma k_B T}{Q}} dW_i,

    where:

    .. math::
        v_i = c_i \\tanh\\left(\\frac{\\alpha_n p_i}{m_i c_i}\\right).

    Here, :math:`c_i = \\sqrt{\\alpha_n n m_i k T}` is speed limit for such degree of freedom and,
    by default, :math:`\\alpha_n = \\frac{n+1}{n}`.
    As usual, the inertial parameter :math:`Q` is defined as :math:`Q = k_B T \\tau^2`, with
    :math:`\\tau` being a relaxation time :cite:`Tuckerman_1992`. An approximate solution is
    obtained by applying the Trotter-Suzuki splitting formula:

    .. math::
        e^{\\delta t\\mathcal{L}} =
        e^{(\\delta t/2)\\mathcal{L}_B}
        e^{(\\delta t/2)\\mathcal{L}_S}
        e^{\\delta t\\mathcal{L}_O}
        e^{(\\delta t/2)\\mathcal{L}_S}
        e^{(\\delta t/2)\\mathcal{L}_B}

    Each exponential operator above is the solution of a differential equation.

    Equation 'B' is a boost, whose solution is:

    .. math::
        v_{\\eta_i}(t) = v_{\\eta_i}^0 + \\frac{1}{Q}\\left(
                         \\frac{n+1}{\\alpha_n n} m_i v_i^2 - k_B T\\right) t

    Equation 'S' is a scaling, whose solution is:

    .. math::
        p_i(t) = \\frac{m_i c_i}{\\alpha_n} \\mathrm{arcsinh}\\left[\\sinh\\left(
                 \\frac{\\alpha_n p_i}{m_i c_i}\\right) e^{-\\alpha_n v_{\\eta_i} t}\\right]

    Equation 'O' is an Ornstein–Uhlenbeck process, whose solution is:

    .. math::
        v_{\\eta_i}(t) = v_{\\eta_i}^0 e^{-\\gamma t}
                   + \\sqrt{\\frac{k_B T}{Q}(1-e^{-2\\gamma t})} R_N

    where :math:`R_N` is a normally distributed random number.

    Parameters
    ----------
        temperature : unit.Quantity
            The temperature of the heat bath.
        n : int or float
            The regulating parameter.
        timeScale : unit.Quantity (time)
            The relaxation time of the Nose-Hoover thermostat.
        frictionConstant : unit.Quantity (1/time)
            The friction coefficient of the Langevin thermostat.

    Keyword args
    ------------
        alpha_n : int or float, default=1
            Another regulating parameter.

    """
    def __init__(self, temperature, n, timeScale, frictionConstant, alpha_n=1, split=False):
        super().__init__()
        self._alpha = alpha_n
        self._n = n
        self._split = split
        self.globalVariables['kT'] = kB*temperature
        self.globalVariables['ankT'] = self._alpha*n*kB*temperature
        self.globalVariables['Q'] = kB*temperature*timeScale**2
        self.globalVariables['omega'] = 1/timeScale
        self.globalVariables['friction'] = frictionConstant
        self.perDofVariables['v_eta'] = 0

    def addSteps(self, integrator, fraction=1.0, force='f'):
        n = self._n
        alpha = self._alpha
        G_definition = f'; G=({(n+1)/(alpha*n)}*m*(c*tanh({alpha}*v/c))^2 - kT)/Q'
        G_definition += '; c=sqrt(ankT/m)'
        boost = f'v_eta + G*{0.5*fraction}*dt' + G_definition
        scaling = f'{1/alpha}*c*asinhz'
        scaling += '; asinhz=(2*step(z)-1)*log(select(step(za-1E8),2*za,za+sqrt(1+z*z))); za=abs(z)'
        scaling += f'; z=sinh({alpha}*v/c)*exp(-v_eta*{0.5*fraction}*dt)'
        scaling += '; c=sqrt(ankT/m)'
        if self._split:
            OU = 'v_eta*z + omega*sqrt(1-z^2)*gaussian'
        else:
            OU = 'v_eta*z + G*(1-z)/friction + omega*sqrt(1-z^2)*gaussian' + G_definition
        OU += f'; z=exp(-friction*{fraction}*dt)'
        self._split and integrator.addComputePerDof('v_eta', boost)
        integrator.addComputePerDof('v', scaling)
        integrator.addComputePerDof('v_eta', OU)
        integrator.addComputePerDof('v', scaling)
        self._split and integrator.addComputePerDof('v_eta', boost)


class RegulatedAtomicNoseHooverLangevinPropagator(Propagator):
    """
    This class implements a regulated version of the massive Nose-Hoover-Langevin propagator
    :cite:`Samoletov_2007,Leimkuhler_2009`. It provides, for every degree of freedom in the system,
    a solution for the following :term:`SDE` system:

    .. math::
        & dp_i = -v_{\\eta_i} p_i dt \\\\
        & dv_{\\eta_i} = \\frac{p_i v_i - k_B T}{Q} dt
                   - \\gamma v_{\\eta_i} dt + \\sqrt{\\frac{2\\gamma k_B T}{Q}}dW_i,

    where:

    .. math::
        v_i = c_i \\tanh\\left(\\frac{\\alpha_n p_i}{m_i c_i}\\right).

    Here, :math:`c_i = \\sqrt{\\frac{\\alpha_n n k_B T}{m_i}}` is the speed limit for such degree
    of freedom and, by default, :math:`\\alpha_n = \\frac{n+1}{n}`.
    As usual, the inertial parameter :math:`Q` is defined as :math:`Q = k_B T \\tau^2`, with
    :math:`\\tau` being a relaxation time :cite:`Tuckerman_1992`. An approximate solution is
    obtained by applying the Trotter-Suzuki splitting formula:

    .. math::
        e^{\\delta t\\mathcal{L}} =
        e^{(\\delta t/2)\\mathcal{L}_B}
        e^{(\\delta t/2)\\mathcal{L}_S}
        e^{\\delta t\\mathcal{L}_O}
        e^{(\\delta t/2)\\mathcal{L}_S}
        e^{(\\delta t/2)\\mathcal{L}_B}

    Each exponential operator above is the solution of a differential equation.

    Equation 'B' is a boost, whose solution is:

    .. math::
        v_{\\eta_i}(t) = v_{\\eta_i}^0 + \\frac{p_i v_i - k_B T}{Q} t

    Equation 'S' is a scaling, whose solution is:

    .. math::
        p_i(t) = p_i^0 e^{-v_{\\eta_i} t}

    Equation 'O' is an Ornstein–Uhlenbeck process, whose solution is:

    .. math::
        v_{\\eta_i}(t) = v_{\\eta_i}^0 e^{-\\gamma t}
                   + \\sqrt{\\frac{k_B T}{Q}(1-e^{-2\\gamma t})} R_{N,i}

    where :math:`R_{N,i}` is a normally distributed random number.

    Parameters
    ----------
        temperature : unit.Quantity
            The temperature of the heat bath.
        n : int or float
            The regulating parameter.
        timeScale : unit.Quantity (time)
            The relaxation time of the Nose-Hoover thermostat.
        frictionConstant : unit.Quantity (1/time)
            The friction coefficient of the Langevin thermostat.

    Keyword args
    ------------
        alpha_n : int or float, default=1
            Another regulating parameter.

    """
    def __init__(self, temperature, n, timeScale, frictionConstant, alpha_n=1, split=False):
        super().__init__()
        self._alpha = alpha_n
        self._split = split
        kT = kB*temperature
        Q = 3*kT*timeScale**2
        self.globalVariables['kT'] = kT
        self.globalVariables['ankT'] = self._alpha*n*kT
        self.globalVariables['Q'] = Q
        self.globalVariables['omega'] = unit.sqrt(kT/Q)
        self.globalVariables['friction'] = frictionConstant
        self.perDofVariables['v_eta'] = 0

    def addSteps(self, integrator, fraction=1.0, force='f'):
        alpha = self._alpha
        G_definition = f'; G=(dot(m*v,c*tanh({alpha}*v/c)) - 3*kT)/Q'
        G_definition += '; c=sqrt(ankT/m)'
        boost = f'v_eta + G*{0.5*fraction}*dt' + G_definition
        scaling = f'v*exp(-v_eta*{0.5*fraction}*dt)'
        if self._split:
            OU = 'v_eta*z + omega*sqrt(1-z^2)*_x(gaussian)'
        else:
            OU = 'v_eta*z + G*(1-z)/friction + omega*sqrt(1-z^2)*_x(gaussian)' + G_definition
        OU += f'; z=exp(-friction*{fraction}*dt)'
        self._split and integrator.addComputePerDof('v_eta', boost)
        integrator.addComputePerDof('v', scaling)
        integrator.addComputePerDof('v_eta', OU)
        integrator.addComputePerDof('v', scaling)
        self._split and integrator.addComputePerDof('v_eta', boost)


class TwiceRegulatedAtomicNoseHooverLangevinPropagator(Propagator):
    """
    This class implements a doubly-regulated version of the atomic Nose-Hoover-Langevin propagator
    :cite:`Samoletov_2007,Leimkuhler_2009`. It provides, for every atom in the system, a solution
    for the following :term:`SDE` system:

    .. math::
        & dv_i = -v_{\\eta,\\mathrm{atom}_i} v_i \\left[
                    1 - \\left(\\frac{v_i}{c_i}\\right)^2\\right] dt \\\\
        & dv_{\\eta,j} = \\frac{1}{Q}\\left(
                    \\frac{n+1}{\\alpha_n n} m_j \\mathbf{v}_j^T \\mathbf{v}_j - 3 k_B T\\right) dt
                - \\gamma v_{\\eta,j} dt + \\sqrt{\\frac{2\\gamma k_B T}{Q}} dW_j,

    where :math:`c_i = \\sqrt{\\alpha_n n m_i k T}` is speed limit for such degree of freedom and,
    by default, :math:`\\alpha_n = \\frac{n+1}{n}`.
    As usual, the inertial parameter :math:`Q` is defined as :math:`Q = 3 k_B T \\tau^2`, with
    :math:`\\tau` being a relaxation time :cite:`Tuckerman_1992`. An approximate solution is
    obtained by applying the Trotter-Suzuki splitting formula:

    .. math::
        e^{\\delta t\\mathcal{L}} =
        e^{(\\delta t/2)\\mathcal{L}_B}
        e^{(\\delta t/2)\\mathcal{L}_S}
        e^{\\delta t\\mathcal{L}_O}
        e^{(\\delta t/2)\\mathcal{L}_S}
        e^{(\\delta t/2)\\mathcal{L}_B}

    Each exponential operator above is the solution of a differential equation.

    Equation 'B' is a boost, whose solution is:

    .. math::
        v_{\\eta,j}(t) = v_{\\eta,j}^0 + \\frac{1}{Q}\\left(
                    \\frac{n+1}{\\alpha_n n} m_j \\mathbf{v}_j^T \\mathbf{v}_j - 3 k_B T\\right) t

    Equation 'S' is a scaling, whose solution is:

    .. math::
        & v_{s,i}(t) = v_i^0 e^{-\\alpha_n v_{\\eta_i} t} \\\\
        & v_i(t) = \\frac{v_{s,i}(t)}{\\sqrt{1 - \\left(\\frac{v_i^0}{c_i}\\right)^2 +
                 \\left(\\frac{v_{s,i}(t)}{c_i}\\right)^2}}

    Equation 'O' is an Ornstein–Uhlenbeck process, whose solution is:

    .. math::
        v_{\\eta_i}(t) = v_{\\eta_i}^0 e^{-\\gamma t}
                   + \\sqrt{\\frac{k_B T}{Q}(1-e^{-2\\gamma t})} R_N

    where :math:`R_N` is a normally distributed random number.

    Parameters
    ----------
        temperature : unit.Quantity
            The temperature of the heat bath.
        n : int or float
            The regulating parameter.
        timeScale : unit.Quantity (time)
            The relaxation time of the Nose-Hoover thermostat.
        frictionConstant : unit.Quantity (1/time)
            The friction coefficient of the Langevin thermostat.

    Keyword args
    ------------
        alpha_n : int or float, default=1
            Another regulating parameter.

    """
    def __init__(self, temperature, n, timeScale, frictionConstant, alpha_n=1, split=False):
        super().__init__()
        self._alpha = alpha_n
        self._n = n
        self._split = split
        Q = 3*kB*temperature*timeScale**2
        self.globalVariables['kT'] = kB*temperature
        self.globalVariables['ankT'] = self._alpha*n*kB*temperature
        self.globalVariables['Q'] = Q
        self.globalVariables['omega'] = unit.sqrt(kB*temperature/Q)
        self.globalVariables['friction'] = frictionConstant
        self.perDofVariables['v_eta'] = 0

    def addSteps(self, integrator, fraction=1.0, force='f'):
        n = self._n
        alpha = self._alpha
        G_definition = f'; G=({(n+1)/(alpha*n)}*dot(m*c*y,c*y) - 3*kT)/Q; y=tanh({alpha}*v/c)'
        G_definition += '; c=sqrt(ankT/m)'
        boost = f'v_eta + G*{0.5*fraction}*dt' + G_definition
        scaling = f'{1/alpha}*c*asinhz'
        scaling += '; asinhz=(2*step(z)-1)*log(select(step(za-1E8),2*za,za+sqrt(1+z*z))); za=abs(z)'
        scaling += f'; z=sinh({alpha}*v/c)*exp(-v_eta*{0.5*fraction}*dt)'
        scaling += '; c=sqrt(ankT/m)'
        if self._split:
            OU = 'v_eta*z + omega*sqrt(1-z^2)*_x(gaussian)'
        else:
            OU = 'v_eta*z + G*(1-z)/friction + omega*sqrt(1-z^2)*_x(gaussian)' + G_definition
        OU += f'; z=exp(-friction*{fraction}*dt)'
        self._split and integrator.addComputePerDof('v_eta', boost)
        integrator.addComputePerDof('v', scaling)
        integrator.addComputePerDof('v_eta', OU)
        integrator.addComputePerDof('v', scaling)
        self._split and integrator.addComputePerDof('v_eta', boost)


class TwiceRegulatedGlobalNoseHooverLangevinPropagator(Propagator):
    """
    This class implements a doubly-regulated version of the global Nose-Hoover-Langevin propagator
    :cite:`Samoletov_2007,Leimkuhler_2009`. It provides, for every degree of freedom in the system,
    a solution for the following :term:`SDE` system:

    .. math::
        & dv_i = -\\alpha_n v_\\eta v_i \\left[1 - \\left(\\frac{v_i}{c_i}\\right)^2\\right] dt \\\\
        & dv_\\eta = \\frac{1}{Q}\\left(\\frac{n+1}{n \\alpha_n} \\mathbf{v}^T \\mathbf{M} \\mathbf{v} -
            N_f k_B T\\right) dt - \\gamma v_\\eta dt + \\sqrt{\\frac{2\\gamma k_B T}{Q}} dW,

    where :math:`c_i = \\sqrt{\\alpha_n n m_i k T}` is speed limit for such degree of freedom and,
    by default, :math:`\\alpha_n = \\frac{n+1}{n}`.
    As usual, the inertial parameter :math:`Q` is defined as :math:`Q = N_f k_B T \\tau^2`, with
    :math:`\\tau` being a relaxation time :cite:`Tuckerman_1992`. An approximate solution is
    obtained by applying the Trotter-Suzuki splitting formula:

    .. math::
        e^{\\delta t\\mathcal{L}} =
        e^{(\\delta t/2)\\mathcal{L}_B}
        e^{(\\delta t/2)\\mathcal{L}_S}
        e^{\\delta t\\mathcal{L}_O}
        e^{(\\delta t/2)\\mathcal{L}_S}
        e^{(\\delta t/2)\\mathcal{L}_B}

    Each exponential operator above is the solution of a differential equation.

    Equation 'B' is a boost, whose solution is:

    .. math::
        v_\\eta(t) = v_\\eta^0 + \\frac{1}{Q}\\left(
            \\frac{n+1}{n \\alpha_n} \\mathbf{v}^T \\mathbf{M} \\mathbf{v}- N_f k_B T\\right) t

    Equation 'S' is a scaling, whose solution is:

    .. math::
        & v_{s,i}(t) = v_i^0 e^{-\\alpha_n v_\\eta t} \\\\
        & v_i(t) = \\frac{v_{s,i}(t)}{\\sqrt{1 - \\left(\\frac{v_i^0}{c_i}\\right)^2 +
                 \\left(\\frac{v_{s,i}(t)}{c_i}\\right)^2}}

    Equation 'O' is an Ornstein–Uhlenbeck process, whose solution is:

    .. math::
        v_\\eta(t) = v_\\eta^0 e^{-\\gamma t}
                   + \\sqrt{\\frac{k_B T}{Q}(1-e^{-2\\gamma t})} R_N

    where :math:`R_N` is a normally distributed random number.

    Parameters
    ----------
        degreesOfFreedom : int
            The number of degrees of freedom in the system
        temperature : unit.Quantity
            The temperature of the heat bath.
        n : int or float
            The regulating parameter.
        timeScale : unit.Quantity (time)
            The relaxation time of the Nose-Hoover thermostat.
        frictionConstant : unit.Quantity (1/time)
            The friction coefficient of the Langevin thermostat.

    Keyword args
    ------------
        alpha_n : int or float, default=1
            Another regulating parameter.

    """
    def __init__(self, degreesOfFreedom, temperature, n, timeScale, frictionConstant, alpha_n=1, split=False):
        super().__init__()
        self._alpha = alpha_n
        self._n = n
        self._split = split
        self._Nf = degreesOfFreedom
        Q = 3*kB*temperature*timeScale**2
        self.globalVariables['kT'] = kB*temperature
        self.globalVariables['ankT'] = self._alpha*n*kB*temperature
        self.globalVariables['Q'] = Q
        self.globalVariables['omega'] = unit.sqrt(kB*temperature/Q)
        self.globalVariables['friction'] = frictionConstant
        self.globalVariables['sum_mvv'] = 0
        self.globalVariables['v_eta'] = 0

    def addSteps(self, integrator, fraction=1.0, force='f'):
        n = self._n
        alpha = self._alpha
        G_definition = f'; G=({(n+1)/(alpha*n)}*sum_mvv - {self._Nf}*kT)/Q'
        boost = f'v_eta + G*{0.5*fraction}*dt' + G_definition
        scaling = f'{1/alpha}*c*asinhz'
        scaling += '; asinhz=(2*step(z)-1)*log(select(step(za-1E8),2*za,za+sqrt(1+z*z))); za=abs(z)'
        scaling += f'; z=sinh({alpha}*v/c)*exp(-v_eta*{0.5*fraction}*dt)'
        scaling += '; c=sqrt(ankT/m)'
        if self._split:
            OU = 'v_eta*z + omega*sqrt(1-z^2)*gaussian'
        else:
            OU = 'v_eta*z + G*(1-z)/friction + omega*sqrt(1-z^2)*gaussian' + G_definition
        OU += f'; z=exp(-friction*{fraction}*dt)'
        if self._split:
            integrator.addComputeSum('sum_mvv', f'm*(c*tanh({alpha}*v/c))^2; c=sqrt(ankT/m)')
            integrator.addComputeGlobal('v_eta', boost)
        integrator.addComputePerDof('v', scaling)
        if not self._split:
            integrator.addComputeSum('sum_mvv', f'm*(c*tanh({alpha}*v/c))^2; c=sqrt(ankT/m)')
        integrator.addComputeGlobal('v_eta', OU)
        integrator.addComputePerDof('v', scaling)
        if self._split:
            integrator.addComputeSum('sum_mvv', f'm*(c*tanh({alpha}*v/c))^2; c=sqrt(ankT/m)')
            integrator.addComputeGlobal('v_eta', boost)
