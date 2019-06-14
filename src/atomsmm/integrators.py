"""
.. module:: integrators
   :platform: Unix, Windows
   :synopsis: a module for defining integrator classes.

.. moduleauthor:: Charlles R. A. Abreu <abreu@eq.ufrj.br>

"""

import functools
import math
import re
import types

import numpy as np
from simtk import openmm
from sympy import Symbol
from sympy.parsing.sympy_parser import parse_expr

import atomsmm.propagators as propagators
from atomsmm.propagators import Propagator as DummyPropagator
from atomsmm.utils import InputError
from atomsmm.utils import kB


class _AtomsMM_Integrator(openmm.CustomIntegrator):
    def __init__(self, stepSize):
        super().__init__(stepSize)
        self.addGlobalVariable('mvv', 0.0)
        self.addGlobalVariable('NDOF', 0.0)
        self.addPerDofVariable('ndof', 0.0)
        self._obsoleteKinetic = True
        self._forceFinder = re.compile('^f[0-9]+$|^f$')
        self._obsoleteContextState = True
        self._random = np.random.RandomState()
        self._uninitialized = True

    def __repr__(self):
        """
        A human-readable version of each integrator step (adapted from openmmtools)

        Returns
        -------
        readable_lines : str
           A list of human-readable versions of each step of the integrator

        """
        step_type_str = [
            '{target} <- {expr}',
            '{target} <- {expr}',
            '{target} <- sum({expr})',
            'constrain positions',
            'constrain velocities',
            'allow forces to update the context state',
            'if ({expr}):',
            'while ({expr}):',
            'end'
        ]
        readable_lines = []
        indent_level = 0
        for step in range(self.getNumComputations()):
            line = ''
            step_type, target, expr = self.getComputationStep(step)
            if step_type == 8:
                indent_level -= 1
            command = step_type_str[step_type].format(target=target, expr=expr)
            line += '{:4d}: '.format(step) + '   '*indent_level + command
            if step_type in [6, 7]:
                indent_level += 1
            readable_lines.append(line)
        return '\n'.join(readable_lines)

    def _normalVec(self):
        return openmm.Vec3(self._random.normal(), self._random.normal(), self._random.normal())

    def _required_variables(self, variable, expression):
        """
        Returns a list of strings containting the names of all global and per-dof variables
        required by an OpenMM CustomIntegrator operation.

        """
        definitions = ('{}={}'.format(variable, expression)).split(';')
        names = set()
        symbols = set()
        for definition in definitions:
            name, expr = definition.split('=')
            names.add(Symbol(name.strip()))
            symbols |= parse_expr(expr.replace('^', '**')).free_symbols
        return list(str(element) for element in (symbols - names))

    def _checkUpdate(self, variable, expression):
        """
        Check whether it is necessary to update the mvv global variable (twice the kinetic energy)
        or to let the openmm.Force objects update the context state.

        """
        requirements = self._required_variables(variable, expression)
        if self._obsoleteKinetic and 'mvv' in requirements:
            super().addComputeSum('mvv', 'm*v*v')
            self._obsoleteKinetic = False
        if self._obsoleteContextState and any(self._forceFinder.match(s) for s in requirements):
            super().addUpdateContextState()
            self._obsoleteContextState = False

    def addUpdateContextState(self):
        if self._obsoleteContextState:
            super().addUpdateContextState()
            self._obsoleteContextState = False

    def addComputeGlobal(self, variable, expression):
        if variable == 'mvv':
            raise InputError('Cannot assign value to global variable mvv')
        self._checkUpdate(variable, expression)
        super().addComputeGlobal(variable, expression)

    def addComputePerDof(self, variable, expression):
        self._checkUpdate(variable, expression)
        super().addComputePerDof(variable, expression)
        if variable == 'v':
            self._obsoleteKinetic = True

    def setRandomNumberSeed(self, seed):
        self._random.seed(seed)
        super().setRandomNumberSeed(self._random.tomaxint() % 2**31)

    def step(self, steps):
        if self._uninitialized:
            ndof = self.getPerDofVariableByName('ndof')
            self._ndof = NDOF = 3*len(ndof)
            self.setGlobalVariableByName('NDOF', NDOF)
            for i in range(len(ndof)):
                ndof[i] = openmm.Vec3(NDOF, NDOF, NDOF)
            self.setPerDofVariableByName('ndof', ndof)
            self.initialize()
            self._uninitialized = False
        return super().step(steps)

    def initialize(self):
        """
        Perform initialization of atomic velocities and other random per-dof variables.

        """
        pass


class GlobalThermostatIntegrator(_AtomsMM_Integrator):
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


class MultipleTimeScaleIntegrator(_AtomsMM_Integrator):
    """
    This class implements a Multiple Time-Scale (MTS) integrator using the RESPA method.

    Parameters
    ----------
        stepSize : unit.Quantity
            The largest time step for numerically integrating the system of equations.
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
    def __init__(self, stepSize, loops, move=None, boost=None, bath=None, **kwargs):
        scheme = kwargs.pop('scheme', 'middle')
        location = kwargs.pop('location', 0)
        nres = kwargs.pop('nres', 1)
        nsy = kwargs.pop('nsy', 1)
        has_memory = kwargs.pop('has_memory', False)
        super().__init__(stepSize)
        if nres > 1:
            bath = propagators.SplitPropagator(bath, nres)
        if nsy > 1:
            bath = propagators.SuzukiYoshidaPropagator(bath, nsy)
        if scheme == 'middle':
            propagator = propagators.RespaPropagator(loops, move=move, boost=boost, core=bath, has_memory=has_memory)
        elif scheme == 'blitz':
            propagator = propagators.BlitzRespaPropagator(loops, move=move, boost=boost, core=bath, has_memory=has_memory)
        elif scheme in ['xi-respa', 'xo-respa', 'side']:
            level = location if scheme == 'side' else (0 if scheme == 'xi-respa' else len(loops)-1)
            propagator = propagators.RespaPropagator(loops, move=move, boost=boost, shell={level: bath}, has_memory=has_memory)
        else:
            raise InputError('wrong value of scheme parameter')
        propagator.addVariables(self)
        propagator.addSteps(self)


class NHL_R_Integrator(MultipleTimeScaleIntegrator):
    """
    This class is an implementation of the massive Nosé-Hoover-Langevin (RESPA) integrator. The
    method consists in solving the following equations for each degree of freedom (DOF) in the
    system:

    .. math::
        & \\frac{dx}{dt} = v \\\\
        & \\frac{dv}{dt} = \\frac{f}{m} - v_2 v \\\\
        & dv_2 = \\frac{m v^2 - kT}{Q_2}dt - \\gamma v_2 dt + \\sqrt{\\frac{2 \\gamma kT}{Q_2}} dW

    The equations are integrated by a reversible, multiple timescale numerical scheme.

    Parameters
    ----------
        stepSize : unit.Quantity
            The largest time step for numerically integrating the system of equations.
        loops : list(int)
            See description in :class:`MultipleTimeScaleIntegrator`.
        temperature : unit.Quantity
            The temperature to which the configurational sampling should correspond.
        timeScale : unit.Quantity
            A time scale :math:`\\tau` from which the inertial parameters are computed as
            :math:`Q_2 = kT\\tau^2`.
        frictionConstant : unit.Quantity
            The friction constant :math:`\\gamma` present in the stochastic equation of motion for
            per-DOF thermostat variable :math:`v_2`.
        **kwargs : keyword arguments
            The same keyword arguments of class :class:`MultipleTimeScaleIntegrator` apply here.

    """
    def __init__(self, stepSize, loops, temperature, timeScale, frictionConstant, **kwargs):
        scaling = propagators.GenericScalingPropagator('v', 'v2')
        DOU = propagators.OrnsteinUhlenbeckPropagator(temperature, frictionConstant,
                                                      'v2', 'Q2', 'm*v^2 - kT',
                                                      Q2=kB*temperature*timeScale**2,
                                                      kT=kB*temperature)
        bath = propagators.TrotterSuzukiPropagator(DOU, scaling)
        super().__init__(stepSize, loops, None, None, bath, **kwargs)

    def initialize(self):
        kT = self.getGlobalVariableByName('kT')
        Q1 = self.getGlobalVariableByName('Q2')
        v2 = self.getPerDofVariableByName('v2')
        S = math.sqrt(kT/Q1)
        for i in range(len(v2)):
            v2[i] = S*self._normalVec()
        self.setPerDofVariableByName('v2', v2)


class Langevin_R_Integrator(MultipleTimeScaleIntegrator):
    """
    This class is an implementation of the multiple time scale Langevin (RESPA) integrator. The
    method consists in solving the following equations for each degree of freedom (DOF) in the
    system:

    .. math::
        & \\frac{dx}{dt} = v \\\\
        & \\frac{dv}{dt} = \\frac{f}{m} - \\gamma v dt + \\sqrt{\\frac{2 \\gamma kT}{m}} dW

    The equations are integrated by a reversible, multiple timescale numerical scheme.

    Parameters
    ----------
        stepSize : unit.Quantity
            The largest time step for numerically integrating the system of equations.
        loops : list(int)
            See description in :class:`MultipleTimeScaleIntegrator`.
        temperature : unit.Quantity
            The temperature to which the configurational sampling should correspond.
        frictionConstant : unit.Quantity
            The friction constant :math:`\\gamma` present in the stochastic equation of motion for
            per-DOF thermostat variable :math:`v_2`.
        **kwargs : keyword arguments
            The same keyword arguments of class :class:`MultipleTimeScaleIntegrator` apply here.

    """
    def __init__(self, stepSize, loops, temperature, frictionConstant, **kwargs):
        bath = propagators.OrnsteinUhlenbeckPropagator(temperature, frictionConstant,
                                                       'v', 'm', kT=kB*temperature)
        super().__init__(stepSize, loops, None, None, bath, **kwargs)


class SIN_R_Integrator(MultipleTimeScaleIntegrator):
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
            The same keyword arguments of class :class:`MultipleTimeScaleIntegrator` apply here.

    """
    def __init__(self, stepSize, loops, temperature, timeScale, frictionConstant, **kwargs):
        L = kwargs.pop('L', 1)
        isoF = propagators.MassiveIsokineticPropagator(temperature, timeScale, L, forceDependent=True)
        isoN = propagators.MassiveIsokineticPropagator(temperature, timeScale, L, forceDependent=False)
        v1 = ['v1_{}'.format(i) for i in range(L)]
        v2 = ['v2_{}'.format(i) for i in range(L)]
        DOU = propagators.OrnsteinUhlenbeckPropagator(temperature, frictionConstant,
                                                      v2[0], 'Q2', 'Q1*{}^2 - kT'.format(v1[0]),
                                                      Q2=kB*temperature*timeScale**2)
        for i in range(1, L):
            New = propagators.OrnsteinUhlenbeckPropagator(temperature, frictionConstant,
                                                          v2[i], 'Q2', 'Q1*{}^2 - kT'.format(v1[i]),
                                                          Q2=kB*temperature*timeScale**2)
            DOU = propagators.ChainedPropagator(DOU, New)
        bath = propagators.TrotterSuzukiPropagator(DOU, isoN)
        super().__init__(stepSize, loops, None, isoF, bath, **kwargs)

    def initialize(self):
        kT = self.getGlobalVariableByName('kT')
        Q1 = self.getGlobalVariableByName('Q1')
        Q2 = self.getGlobalVariableByName('Q2')
        L = round(self.getGlobalVariableByName('L'))
        for i in range(L):
            v1 = self.getPerDofVariableByName('v1_{}'.format(i))
            v2 = self.getPerDofVariableByName('v2_{}'.format(i))
            S1 = math.sqrt((L + 1)/L*kT/Q1)
            S2 = math.sqrt(kT/Q2)
            for j in range(len(v1)):
                v1[j] = S1*self._normalVec()
                v2[j] = S2*self._normalVec()
            self.setPerDofVariableByName('v1_{}'.format(i), v1)
            self.setPerDofVariableByName('v2_{}'.format(i), v2)


class NewMethodIntegrator(MultipleTimeScaleIntegrator):
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
            The same keyword arguments of class :class:`MultipleTimeScaleIntegrator` apply here.

    """
    def __init__(self, stepSize, loops, temperature, timeScale, frictionConstant, **kwargs):
        L = kwargs.pop('L', 1)
        self._massive = kwargs.pop('massive', True)
        newF = propagators.NewMethodPropagator(temperature, timeScale, L, forceDependent=True)
        newN = propagators.NewMethodPropagator(temperature, timeScale, L, forceDependent=False)
        mass = 'Q_eta' if self._massive else 'NDOF*Q_eta'
        force = ('{}*m*v*v - kT' if self._massive else '{}*mvv - NDOF*kT').format((L+1)/L)
        DOU = propagators.OrnsteinUhlenbeckPropagator(temperature, frictionConstant,
                                                      'v_eta', mass, force,
                                                      overall=(not self._massive),
                                                      Q_eta=L*kB*temperature*timeScale**2)
        bath = propagators.TrotterSuzukiPropagator(DOU, newN)
        super().__init__(stepSize, loops, None, newF, bath, **kwargs)

    def initialize(self):
        kT = self.getGlobalVariableByName('kT')
        Q_eta = self.getGlobalVariableByName('Q_eta')
        if self._massive:
            sigma = math.sqrt(kT/Q_eta)
            v_eta = self.getPerDofVariableByName('v_eta')
            for i in range(len(v_eta)):
                v_eta[i] = sigma*self._normalVec()
            self.setPerDofVariableByName('v_eta', v_eta)
        else:
            sigma = math.sqrt(kT/(self._ndof*Q_eta))
            self.setGlobalVariableByName('v_eta', sigma*self._random.normal())


class LimitedSpeedBAOABIntegrator(MultipleTimeScaleIntegrator):
    """

    Parameters
    ----------
        stepSize : unit.Quantity
            The largest time step for numerically integrating the system of equations.
        loops : list(int)
            See description in :class:`MultipleTimeScaleIntegrator`.
        temperature : unit.Quantity
            The temperature to which the configurational sampling should correspond.
        frictionConstant : unit.Quantity
            The friction constant :math:`\\gamma` present in the stochastic equation of motion for
            per-DOF thermostat variable :math:`v_2`.
        **kwargs : keyword arguments
            The same keyword arguments of class :class:`MultipleTimeScaleIntegrator` apply here.

    """
    def __init__(self, stepSize, loops, temperature, frictionConstant, **kwargs):
        L = kwargs.pop('L', 1)
        move = propagators.LimitedSpeedLangevinPropagator(temperature, frictionConstant, L, 'move')
        boost = propagators.LimitedSpeedLangevinPropagator(temperature, frictionConstant, L, 'boost')
        bath = propagators.LimitedSpeedLangevinPropagator(temperature, frictionConstant, L, 'bath')
        super().__init__(stepSize, loops, move, boost, bath, **kwargs)
        # self.addComputePerDof('v', 'sqrt(LkT*p*tanh(p)/m)')
        self.addComputePerDof('v', 'sqrt(LkT/m)*tanh(p)')
        # self.addComputePerDof('v', 'p')

    def initialize(self):
        # Implement initial values of per-dof variable p
        pass


class LimitedSpeedNHLIntegrator(MultipleTimeScaleIntegrator):
    """

    Parameters
    ----------
        stepSize : unit.Quantity
            The largest time step for numerically integrating the system of equations.
        loops : list(int)
            See description in :class:`MultipleTimeScaleIntegrator`.
        temperature : unit.Quantity
            The temperature to which the configurational sampling should correspond.
        frictionConstant : unit.Quantity
            The friction constant :math:`\\gamma` present in the stochastic equation of motion for
            per-DOF thermostat variable :math:`v_2`.
        **kwargs : keyword arguments
            The same keyword arguments of class :class:`MultipleTimeScaleIntegrator` apply here.

    """
    def __init__(self, stepSize, loops, temperature, timeScale, frictionConstant, **kwargs):
        L = kwargs.pop('L', 1)
        move = propagators.LimitedSpeedNHLPropagator(temperature, timeScale, frictionConstant, L, 'move')
        boost = propagators.LimitedSpeedNHLPropagator(temperature, timeScale, frictionConstant, L, 'boost')
        bath = propagators.LimitedSpeedNHLPropagator(temperature, timeScale, frictionConstant, L, 'bath')
        super().__init__(stepSize, loops, move, boost, bath, **kwargs)
        # self.addComputePerDof('v', 'sqrt(LkT*p*tanh(p)/m)')
        self.addComputePerDof('v', 'sqrt(LkT/m)*tanh(p)')
        # self.addComputePerDof('v', 'p')

    def initialize(self):
        kT = self.getGlobalVariableByName('kT')
        Q_eta = self.getGlobalVariableByName('Q_eta')
        sigma = math.sqrt(kT/Q_eta)
        v_eta = self.getPerDofVariableByName('v_eta')
        for i in range(len(v_eta)):
            v_eta[i] = sigma*self._normalVec()
        self.setPerDofVariableByName('v_eta', v_eta)
        p = self.getPerDofVariableByName('p')
        for i in range(len(v_eta)):
            p[i] = self._normalVec()
        self.setPerDofVariableByName('p', p)


class LimitedSpeedStochasticIntegrator(MultipleTimeScaleIntegrator):
    """

    Parameters
    ----------
        stepSize : unit.Quantity
            The largest time step for numerically integrating the system of equations.
        loops : list(int)
            See description in :class:`MultipleTimeScaleIntegrator`.
        temperature : unit.Quantity
            The temperature to which the configurational sampling should correspond.
        frictionConstant : unit.Quantity
            The friction constant :math:`\\gamma` present in the stochastic equation of motion for
            per-DOF thermostat variable :math:`v_2`.
        **kwargs : keyword arguments
            The same keyword arguments of class :class:`MultipleTimeScaleIntegrator` apply here.

    """
    def __init__(self, stepSize, loops, temperature, timeScale, frictionConstant, **kwargs):
        L = kwargs.pop('L', 1)
        move = propagators.LimitedSpeedStochasticPropagator(temperature, timeScale, frictionConstant, L, 'move')
        boost = propagators.LimitedSpeedStochasticPropagator(temperature, timeScale, frictionConstant, L, 'boost')
        bath = propagators.LimitedSpeedStochasticPropagator(temperature, timeScale, frictionConstant, L, 'bath')
        super().__init__(stepSize, loops, move, boost, bath, **kwargs)
        # self.addComputePerDof('v', 'sqrt(LkT*p*tanh(p)/m)')
        self.addComputePerDof('v', 'sqrt(LkT/m)*tanh(p)')
        # self.addComputePerDof('v', 'p')

    def initialize(self):
        kT = self.getGlobalVariableByName('kT')
        Q_eta = self.getGlobalVariableByName('Q_eta')
        sigma = math.sqrt(kT/Q_eta)
        v_eta = self.getPerDofVariableByName('v_eta')
        for i in range(len(v_eta)):
            v_eta[i] = sigma*self._normalVec()
        self.setPerDofVariableByName('v_eta', v_eta)
        p = self.getPerDofVariableByName('p')
        for i in range(len(v_eta)):
            p[i] = self._normalVec()
        self.setPerDofVariableByName('p', p)


class LimitedSpeedStochasticVelocityIntegrator(MultipleTimeScaleIntegrator):
    """

    Parameters
    ----------
        stepSize : unit.Quantity
            The largest time step for numerically integrating the system of equations.
        loops : list(int)
            See description in :class:`MultipleTimeScaleIntegrator`.
        temperature : unit.Quantity
            The temperature to which the configurational sampling should correspond.
        frictionConstant : unit.Quantity
            The friction constant :math:`\\gamma` present in the stochastic equation of motion for
            per-DOF thermostat variable :math:`v_2`.
        **kwargs : keyword arguments
            The same keyword arguments of class :class:`MultipleTimeScaleIntegrator` apply here.

    """
    def __init__(self, stepSize, loops, temperature, timeScale, frictionConstant, **kwargs):
        L = kwargs.pop('L', 1)
        move = propagators.LimitedSpeedStochasticVelocityPropagator(temperature, timeScale, frictionConstant, L, 'move')
        boost = propagators.LimitedSpeedStochasticVelocityPropagator(temperature, timeScale, frictionConstant, L, 'boost')
        bath = propagators.LimitedSpeedStochasticVelocityPropagator(temperature, timeScale, frictionConstant, L, 'bath')
        super().__init__(stepSize, loops, move, boost, bath, **kwargs)

    def initialize(self):
        kT = self.getGlobalVariableByName('kT')
        Q_eta = self.getGlobalVariableByName('Q_eta')
        sigma = math.sqrt(kT/Q_eta)
        v_eta = self.getPerDofVariableByName('v_eta')
        for i in range(len(v_eta)):
            v_eta[i] = sigma*self._normalVec()
        self.setPerDofVariableByName('v_eta', v_eta)


class AdiabaticFreeEnergyDynamicsIntegrator(_AtomsMM_Integrator):
    """
    This class implements the Adiabatic Free Energy Dynamics (AFED) method.

    .. math::
        e^{2 n \\Delta t \\mathcal{L}} =
            [e^{\\delta t \\mathcal{L}_\\mathrm{NVT}}]^n
            e^{2 n \\Delta t \\mathcal{L}_\\mathrm{AFED}}
            [e^{\\delta t \\mathcal{L}_\\mathrm{NVT}}]^n

    Parameters
    ----------
        nvt_integrator : openmm.CustomIntegrator
            The NVT integrator.
        steps : int
            The number of NVT steps.
        variable : str
            The name of the extended-space variable.
        mass : Number or unit.Quantity
            The mass of the extended-space variable.
        kT : Number of unit.Quantity
            The temperature of the extended-space variable.
        **kwargs : keyword arguments
            Keyword arguments.

    """
    _v = '_v_extra'
    _m = '_m_extra'
    _kT = '_kT_extra'
    _v_eta = '_v_eta_extra'
    _Q_eta = '_Q_eta_extra'
    _counter = '_nvt_steps_counter'

    def __init__(self, nvt_integrator, steps, variable, mass, kT,
                 time_scale=None, lower_limit=0, upper_limit = 1, wall_stiffness=100000000):
        step_size = 2*steps*nvt_integrator.getStepSize()
        tau = 10*step_size if time_scale is None else time_scale

        super().__init__(step_size)

        self._x = variable
        self._lower_limit = lower_limit
        self._upper_limit = upper_limit
        self._K_wall = wall_stiffness
        self._initialize_function = self._copy_function(nvt_integrator.initialize)

        self.addGlobalVariable(self._v, 0)
        self.addGlobalVariable(self._m, mass)
        self.addGlobalVariable(self._kT, kT)
        self.addGlobalVariable(self._v_eta, 0)
        self.addGlobalVariable(self._Q_eta, kT*tau**2)
        if steps > 1:
            self.addGlobalVariable(self._counter, 0)

        self._import_variables(nvt_integrator, steps)

        self._add_physical_steps(nvt_integrator, steps)
        self._add_extended_space_steps()
        self._add_physical_steps(nvt_integrator, steps)

    def _add_extended_space_steps(self):
        move = '{} + 0.5*dt*{}'.format(self._x, self._v)
        kick = '{} + 0.5*dt*({}*{}^2-{})/{}'.format(self._v_eta, self._m, self._v, self._kT, self._Q_eta)
        scale = '{}*exp(-dt*{})'.format(self._v, self._v_eta)
        self.addComputeGlobal(self._x, move)
        self.addComputeGlobal(self._v_eta, kick)
        self.addComputeGlobal(self._v, scale)
        self.addComputeGlobal(self._v_eta, kick)
        self.addComputeGlobal(self._x, move)

    def _add_physical_steps(self, integrator, steps):
        if steps > 1:
            self.addComputeGlobal(self._counter, '0')
            self.beginWhileBlock('{} < {}'.format(self._counter, steps))
        lower = ' - {}*step(y)*y'.format(self._K_wall) if self._lower_limit is not None else ''
        upper = ' + {}*step(z)*z'.format(self._K_wall) if self._upper_limit is not None else ''
        boost = '{} - {}*dt*dEdx/{}'.format(self._v, 0.25/steps, self._m)
        boost += '; dEdx = deriv(energy,{}){}{}'.format(self._x, upper, lower)
        if self._lower_limit is not None:
            boost += '; y = ({}) - {}'.format(self._lower_limit, self._x)
        if self._upper_limit is not None:
            boost += '; z = {} - ({})'.format(self._x, self._upper_limit)
        self.addComputeGlobal(self._v, boost)
        self._import_computations(integrator, steps)
        self.addComputeGlobal(self._v, boost)
        if steps > 1:
            self.addComputeGlobal(self._counter, '{} + 1'.format(self._counter))
            self.endBlock()

    @staticmethod
    def _copy_function(f):
        g = types.FunctionType(f.__code__, f.__globals__)
        return functools.update_wrapper(g, f)

    def _import_computations(self, integrator, steps):
        regex = re.compile(r'\bdt\b')
        for index in range(integrator.getNumComputations()):
            computation, variable, expression = integrator.getComputationStep(index)
            expression = regex.sub('(dt/{})'.format(2*steps), expression)
            if computation == openmm.CustomIntegrator.ComputeGlobal:
                self.addComputeGlobal(variable, expression)
            elif computation == openmm.CustomIntegrator.ComputePerDof:
                self.addComputePerDof(variable, expression)
            elif computation == openmm.CustomIntegrator.ComputeSum:
                self.addComputeSum(variable, expression)
            elif computation == openmm.CustomIntegrator.ConstrainPositions:
                self.addConstrainPositions()
            elif computation == openmm.CustomIntegrator.ConstrainVelocities:
                self.addConstrainVelocities()
            elif computation == openmm.CustomIntegrator.UpdateContextState:
                self.addUpdateContextState()
            elif computation == openmm.CustomIntegrator.IfBlockStart:
                self.beginIfBlock(expression)
            elif computation == openmm.CustomIntegrator.WhileBlockStart:
                self.beginWhileBlock(expression)
            elif computation == openmm.CustomIntegrator.BlockEnd:
                self.endBlock()

    def _import_variables(self, integrator, steps):
        for index in range(integrator.getNumGlobalVariables()):
            name = integrator.getGlobalVariableName(index)
            if name not in ['mvv', 'NDOF']:
                value = integrator.getGlobalVariable(index)
                self.addGlobalVariable(name, value)
        for index in range(integrator.getNumPerDofVariables()):
            name = integrator.getPerDofVariableName(index)
            if name != 'ndof':
                values = integrator.getPerDofVariable(index)
                self.addPerDofVariable(name, 0)
                self.setPerDofVariableByName(name, values)

    def initialize(self):
        self._initialize_function(self)
