"""
.. module:: integrators
   :platform: Unix, Windows
   :synopsis: a module for defining integrator classes.

.. moduleauthor:: Charlles R. A. Abreu <abreu@eq.ufrj.br>

"""

import math
import re

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
        self._forceFinder = re.compile('f[0-9]*')
        self._obsoleteContextState = True
        self._random = np.random.RandomState()
        self._uninitialized = True

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
            NDOF = 3*len(ndof)
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
        super().__init__(stepSize)
        if nres > 1:
            bath = propagators.SplitPropagator(bath, nres)
        if nsy > 1:
            bath = propagators.SuzukiYoshidaPropagator(bath, nsy)
        if scheme == 'middle':
            propagator = propagators.RespaPropagator(loops, move=move, boost=boost, core=bath)
        elif scheme == 'blitz':
            propagator = propagators.BlitzRespaPropagator(loops, move=move, boost=boost, core=bath)
        elif scheme in ['xi-respa', 'xo-respa', 'side']:
            level = location if scheme == 'side' else (0 if scheme == 'xi-respa' else len(loops)-1)
            propagator = propagators.RespaPropagator(loops, move=move, boost=boost, shell={level: bath})
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
        isoF = propagators.MassiveIsokineticPropagator(temperature, timeScale, forceDependent=True)
        isoN = propagators.MassiveIsokineticPropagator(temperature, timeScale, forceDependent=False)
        DOU = propagators.OrnsteinUhlenbeckPropagator(temperature, frictionConstant, 'v2', 'Q2', 'Q1*v1*v1 - kT')
        bath = propagators.TrotterSuzukiPropagator(DOU, isoN)
        super().__init__(stepSize, loops, None, isoF, bath, **kwargs)

    def initialize(self):
        kT = self.getGlobalVariableByName('kT')
        Q1 = self.getGlobalVariableByName('Q1')
        Q2 = self.getGlobalVariableByName('Q2')
        v1 = self.getPerDofVariableByName('v1')
        v2 = self.getPerDofVariableByName('v2')
        S1 = math.sqrt(2*kT/Q1)
        S2 = math.sqrt(kT/Q2)
        for i in range(len(v1)):
            v1[i] = S1*self._normalVec()
            v2[i] = S2*self._normalVec()
        self.setPerDofVariableByName('v1', v1)
        self.setPerDofVariableByName('v2', v2)


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
        self.L = kwargs.pop('L', 1)
        self.massive = kwargs.pop('massive', False)
        self.kT = kB*temperature

        newF = propagators.NewMethodPropagator(temperature, timeScale, self.L, forceDependent=True)
        newN = propagators.NewMethodPropagator(temperature, timeScale, self.L, forceDependent=False)

        mass = 'Q_eta' if self.massive else 'NDOF*Q_eta'
        force = '(L+one)*m*v*v/L - kT' if self.massive else '(L+1)*mvv/L - NDOF*kT'
        DOU = propagators.OrnsteinUhlenbeckPropagator(temperature, frictionConstant,
                                                      'v_eta', mass, force,
                                                      overall=(not self.massive),
                                                      Q_eta=self.kT*timeScale**2)

        bath = propagators.TrotterSuzukiPropagator(DOU, newN)
        super().__init__(stepSize, loops, None, newF, bath, **kwargs)

    def initialize(self):
        kT = self.getGlobalVariableByName('kT')

        pi = self.getPerDofVariableByName('pi')
        sigma_tanh_pi = 1/np.sqrt(self.L + 1)
        sigma = 2*sigma_tanh_pi**(3/2)  # EMPIRICAL: SHOULD BE MODIFIED
        for i in range(len(pi)):
            pi[i] = sigma*self._normalVec()
        self.setPerDofVariableByName('pi', pi)

        Q_eta = self.getGlobalVariableByName('Q_eta')
        if self.massive:
            sigma = math.sqrt(kT/Q_eta)
            v_eta = self.getPerDofVariableByName('v_eta')
            for i in range(len(v_eta)):
                v_eta[i] = sigma*self._normalVec()
            self.setPerDofVariableByName('v_eta', v_eta)
        else:
            sigma = math.sqrt(kT/(3*len(pi)*Q_eta))
            self.setGlobalVariableByName('v_eta', sigma*self._random.normal())
