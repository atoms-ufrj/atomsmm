"""
.. module:: forces
   :platform: Unix, Windows
   :synopsis: a module for defining the basis class :class:`Force` and derived classes.

.. moduleauthor:: Charlles R. A. Abreu <abreu@eq.ufrj.br>

.. _CustomBondForce: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.CustomBondForce.html
.. _CustomNonbondedForce: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.CustomNonbondedForce.html
.. _Force: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.Force.html
.. _NonbondedForce: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.NonbondedForce.html
.. _System: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.System.html

"""

import math

from simtk import openmm
from simtk import unit

import atomsmm


class _AtomsMM_Force:
    """
    This is the base class of every AtomsMM Force object. In AtomsMM, a force object is a
    combination of OpenMM Force_ objects treated as a single force.

    Parameters
    ----------
        forces : list(openmm.Force)
            A list of OpenMM Force objects.

    """
    def addTo(self, system):
        system.addForce(self)


class _AtomsMM_CompoundForce:
    """
    This is the base class of every AtomsMM Force object. In AtomsMM, a force object is a
    combination of OpenMM Force_ objects treated as a single force.

    Parameters
    ----------
        forces : list(openmm.Force)
            A list of OpenMM Force objects.

    """
    def __init__(self, forces):
        self.forces = forces if isinstance(forces, list) else [forces]
        self.setForceGroup(0)

    def __iter__(self):
        self.current = 0
        return self

    def __next__(self):
        if self.current == len(self.forces):
            raise StopIteration
        else:
            self.current += 1
            return self.forces[self.current - 1]

    def __getitem__(self, i):
        return self.forces[i]

    def __getattr__(self, method):
        """
        Returns a closure which, in turn, will apply the `method`, with the same arguments of
        the original call, to all OpenMM forces which are stored in `self` and have `method` as an
        attribute. In the end, the closure will return `self` for the purpose of chaining.
        """
        def closure(*args, **kwargs):
            for instance in self.forces:
                if hasattr(instance, method):
                    getattr(instance, method)(*args, **kwargs)
            return self
        return closure

    def getForceGroup(self):
        """
        Retrieve the force group to which this :class:`Force` object belongs.

        Returns
        -------
            int
                The group index. Acceptable values lie between 0 and 31 (inclusive).

        """
        return self.forces[0].getForceGroup()

    def addTo(self, system):
        """
        Add the :class:`Force` object to an OpenMM System_.

        Parameters
        ----------
            system : openmm.System
                The system to which the force is being added.

        Returns
        -------
            :class:`Force`
                The object is returned for chaining purposes.

        """
        for force in self.forces:
            system.addForce(force)
        return self

    def enableExceptions(self):
        """
        Enables the :class:`Force` object to incorporate non-exclusion exceptions from an external
        Force_ when their parameters are imported via :func:`~Force.importFrom`. By default, only
        exclusion exceptions are incorporated.

        In an OpenMM Force_ object, exceptions consist of atom pairs whose interaction parameters
        are determined individually rather than by general like-atom interactions and unlike-atom
        mixing rules. Among them, exclusion exceptions are atom pairs which do not interact at all.

        Returns
        -------
            :class:`Force`
                The object is returned for chaining purposes.

        """
        exceptions = NonbondedExceptionsForce()
        exceptions.setForceGroup(self.getForceGroup())
        self.forces.append(exceptions)
        return self


class _AtomsMM_NonbondedForce(openmm.NonbondedForce, _AtomsMM_Force):
    """
    An extension of OpenMM's NonbondedForce_ class, but without non-exclusion exceptions. These must
    be handled separately using a :class:`_AtomsMM_CustomBondForce` object.

    .. warning::
        When the :func:`~_AtomsMM_NonbondedForce.importFrom` is used, all exceptions of the OpenMM
        NonbondedForce_ passed as argument will be turned into exclusions.

    Parameters
    ----------
        cutoff_distance : Number or unit.Quantity
            The cutoff distance being used for nonbonded interactions.
        switch_distance : Number or unit.Quantity, optional, default=None
            The distance at which the switching function begins to reduce the interaction. If this
            is None, then no switching will be done.

    """
    def __init__(self, cutoff_distance, switch_distance=None):
        super().__init__()
        self.setCutoffDistance(cutoff_distance)
        if switch_distance is None:
            self.setUseSwitchingFunction(False)
        else:
            self.setUseSwitchingFunction(True)
            self.setSwitchingDistance(switch_distance)

    def importFrom(self, force):
        """
        Import all particles and all exceptions from a passed OpenMM NonbondedForce_ object, but
        transforming the non-exclusion exceptions into exclusion ones. Also imports the employed
        nonbondedMethod, PME parameters and Ewald error tolerance, as well as whether to use
        dispersion correction or not.

        Parameters
        ----------
            force : openmm.NonbondedForce
                The force from which the particles and exclusions will be imported.

        Returns
        -------
            :class:`_AtomsMM_NonbondedForce`
                The object is returned for chaining purposes.

        """
        for index in range(force.getNumParticles()):
            self.addParticle(*force.getParticleParameters(index))
        for index in range(force.getNumExceptions()):
            i, j, chargeprod, sigma, epsilon = force.getExceptionParameters(index)
            self.addException(i, j, 0.0*chargeprod, sigma, 0.0*epsilon)

        self.setNonbondedMethod(force.getNonbondedMethod())
        self.setEwaldErrorTolerance(force.getEwaldErrorTolerance())
        self.setPMEParameters(*force.getPMEParameters())
        self.setUseDispersionCorrection(force.getUseDispersionCorrection())

        return self


class _AtomsMM_CustomNonbondedForce(openmm.CustomNonbondedForce, _AtomsMM_Force):
    """
    An extension of OpenMM's CustomNonbondedForce_ class.

    Parameters
    ----------
        energy : str
            An algebraic expression for the interaction energy between two particles as a function
            of their distance `r`, as well as per-particle and global parameters.
        cutoff_distance : unit.Quantity
            The cutoff distance to be used for nonbonded interactions. If this is `None`, then the
            cutoff distance will be imported from an NonbondedForce_ object when the method
            :func:`~_AtomsMM_CustomNonbondedForce.importFrom` is used.
        switch_distance : unit.Quantity, optional, default=None
            In the range from the switch distance to the cutoff distance, a 5th degree spline is
            multiplied by the potential energy in order to make it smoothly decay to zero. If this
            is `None`, then such multiplication will not take place.
        use_dispersion_correction : bool, optional, default=None
            Whether to apply long-range dispersion correction during force and energy calculations.
            If this is `None`, then the decision will be made when using the method
            :func:`~_AtomsMM_CustomNonbondedForce.importFrom`, based on the status of the passed
            NonbondedForce_.

    Keyword Args
    ----------
        parameters : unit.Quantity
            Default values for all global parameters present in `energy` must be passed as keyword
            arguments.

    """
    def __init__(self, energy, cutoff_distance=None, use_switching_function=None,
                 switch_distance=None, use_dispersion_correction=None, **global_parameters):
        super().__init__(energy)
        self.importCutoffDistance = cutoff_distance is None
        self.importUseSwitchingFunction = use_switching_function is None
        self.importSwitchDistance = switch_distance is None
        self.importUseDispersionCorrection = use_dispersion_correction is None
        for name, value in global_parameters.items():
            self.addGlobalParameter(name, value)
        for parameter in ['charge', 'sigma', 'epsilon']:
            self.addPerParticleParameter(parameter)
        if not self.importCutoffDistance:
            self.setCutoffDistance(cutoff_distance)
        if not self.importUseSwitchingFunction:
            self.setUseSwitchingFunction(use_switching_function)
        if not self.importSwitchDistance:
            self.setSwitchingDistance(switch_distance)
        if not self.importUseDispersionCorrection:
            self.setUseLongRangeCorrection(use_dispersion_correction)

    def __repr__(self):
        terms = [s.strip(' \t') for s in self.getEnergyFunction().split(';')]
        return '\n'.join(terms)

    def mixingRules(self, offset_parameters):
        expr1 = dict(charge='charge1', sigma='sigma1', epsilon='epsilon1')
        expr2 = dict(charge='charge2', sigma='sigma2', epsilon='epsilon2')
        for parameter in offset_parameters:
            for property in ['charge', 'sigma', 'epsilon']:
                scale = '{}Scale_{}'.format(property, parameter)
                expr1[property] += '+{}*{}1'.format(parameter, scale)
                expr2[property] += '+{}*{}2'.format(parameter, scale)
        mixing_rules = ';chargeprod = ({})*({})'.format(expr1['charge'], expr2['charge'])
        mixing_rules += ';sigma = 0.5*({}+{})'.format(expr1['sigma'], expr2['sigma'])
        mixing_rules += ';epsilon = sqrt(({})*({}))'.format(expr1['epsilon'], expr2['epsilon'])
        return mixing_rules

    def importFrom(self, nonbonded):
        """
        Import features from a passed NonbondedForce_ object, including all particles and
        exceptions. All non-exclusion exceptions are turned into exclusion ones.

        Parameters
        ----------
            nonbonded : openmm.NonbondedForce
                The force from which to import features.

        Returns
        -------
            :class:`_AtomsMM_CustomNonbondedForce`
                The object itself is returned for chaining purposes.

        """
        builtin = openmm.NonbondedForce
        custom = openmm.CustomNonbondedForce
        mapping = {builtin.CutoffNonPeriodic: custom.CutoffNonPeriodic,
                   builtin.CutoffPeriodic: custom.CutoffPeriodic,
                   builtin.Ewald: custom.CutoffPeriodic,
                   builtin.NoCutoff: custom.NoCutoff,
                   builtin.PME: custom.CutoffPeriodic}
        self.setNonbondedMethod(mapping[nonbonded.getNonbondedMethod()])
        if self.importCutoffDistance:
            self.setCutoffDistance(nonbonded.getCutoffDistance())
        if self.importUseSwitchingFunction:
            self.setUseSwitchingFunction(nonbonded.getUseSwitchingFunction())
        if self.importSwitchDistance:
            self.setSwitchingDistance(nonbonded.getSwitchingDistance())
        if self.importUseDispersionCorrection:
            self.setUseLongRangeCorrection(nonbonded.getUseDispersionCorrection())
        offset_parameters = atomsmm.utils.particleOffsetParameters(nonbonded)
        self.setEnergyFunction(self.getEnergyFunction() + self.mixingRules(offset_parameters))
        for parameter, value in offset_parameters.items():
            self.addGlobalParameter(parameter, value)
            for property in ['charge', 'sigma', 'epsilon']:
                self.addPerParticleParameter('{}Scale_{}'.format(property, parameter))
        extra = [0.0]*3*len(offset_parameters)
        for i in range(nonbonded.getNumParticles()):
            originals = nonbonded.getParticleParameters(i)
            self.addParticle(originals + extra)
        position = dict((name, k+1) for k, name in enumerate(offset_parameters))
        for index in range(nonbonded.getNumParticleParameterOffsets()):
            parameter, particleIndex, chargeScale, sigmaScale, epsilonScale = nonbonded.getParticleParameterOffset(index)
            stored = self.getParticleParameters(particleIndex)
            values = list(stored)
            start = 3*position[parameter]
            values[start:start+3] = [chargeScale, sigmaScale, epsilonScale]
            self.setParticleParameters(particleIndex, values)
        for index in range(nonbonded.getNumExceptions()):
            i, j, _, _, _ = nonbonded.getExceptionParameters(index)
            self.addExclusion(i, j)
        return self

    def getGlobalParameters(self):
        """
        Return a dictionary with all global parameters and their default values.

        """
        globals = dict()
        for index in range(self.getNumGlobalParameters()):
            globals[self.getGlobalParameterName(index)] = self.getGlobalParameterDefaultValue(index)
        return globals


class _AtomsMM_CustomBondForce(openmm.CustomBondForce, _AtomsMM_Force):
    """
    An extension of OpenMM's CustomNonbondedForce_ class.

    Parameters
    ----------
        energy : str
            An algebraic expression giving the interaction energy between two particles as a
            function of their distance `r`, as well as any per-particle and global parameters.

    """
    def __init__(self, energy, **globalParams):
        super().__init__(energy)
        for name, value in globalParams.items():
            self.addGlobalParameter(name, value)

    def offsetRules(self, offset_parameters):
        expr = dict(chargeprod='chargeprod0', sigma='sigma0', epsilon='epsilon0')
        for parameter in offset_parameters:
            for property in ['chargeprod', 'sigma', 'epsilon']:
                scale = '{}Scale_{}'.format(property, parameter)
                expr[property] += '+{}*{}'.format(parameter, scale)
        offset_rules = ';chargeprod = {}'.format(expr['chargeprod'])
        offset_rules += ';sigma = {}'.format(expr['sigma'])
        offset_rules += ';epsilon = {}'.format(expr['epsilon'])
        return offset_rules

    def importFrom(self, nonbonded, extract=False):
        """
        Import all non-exclusion exceptions from the a passed OpenMM NonbondedForce_ object.

        Parameters
        ----------
            force : openmm.NonbondedForce
                The force from which the exceptions will be imported.
            extract : bool, optional, default=False
                Whether the imported exceptions should be removed from force.

        Returns
        -------
            :class:`Force`
                The object is returned for chaining purposes.

        """
        self.setUsesPeriodicBoundaryConditions(nonbonded.usesPeriodicBoundaryConditions())
        offset_parameters = atomsmm.utils.exceptionOffsetParameters(nonbonded)
        if offset_parameters:
            self.setEnergyFunction(self.getEnergyFunction() + self.offsetRules(offset_parameters))
            for property in ['chargeprod', 'sigma', 'epsilon']:
                self.addPerBondParameter('{}0'.format(property))
            for parameter, value in offset_parameters.items():
                self.addGlobalParameter(parameter, value)
                for property in ['chargeprod', 'sigma', 'epsilon']:
                    self.addPerBondParameter('{}Scale_{}'.format(property, parameter))
        else:
            for property in ['chargeprod', 'sigma', 'epsilon']:
                self.addPerBondParameter(property)
        extra = [0.0]*3*len(offset_parameters)
        for index in range(nonbonded.getNumExceptions()):
            i, j, chargeprod, sigma, epsilon = nonbonded.getExceptionParameters(index)
            self.addBond(i, j, [chargeprod, sigma, epsilon] + extra)
            if extract:
                nonbonded.setExceptionParameters(index, i, j, 0.0, 1.0, 0.0)
        position = dict((name, k+1) for k, name in enumerate(offset_parameters))
        for index in range(nonbonded.getNumExceptionParameterOffsets()):
            parameter, bondIndex, chargeprodScale, sigmaScaleScale, epsilonScale = nonbonded.getExceptionParameterOffset(index)
            i, j, stored = self.getBondParameters(bondIndex)
            values = list(stored)
            start = 3*position[parameter]
            values[start:start+3] = [chargeprodScale, sigmaScaleScale, epsilonScale]
            self.setBondParameters(bondIndex, i, j, values)
        return self


class NonbondedExceptionsForce(_AtomsMM_CustomBondForce):
    """
    A special class designed to compute only the exceptions of an OpenMM NonbondedForce object.

    """
    def __init__(self):
        super().__init__('4*epsilon*x*(x-1) + Kc*chargeprod/r; x=(sigma/r)^6',
                         Kc=138.935456*unit.kilojoules_per_mole/unit.nanometer)


class DampedSmoothedForce(_AtomsMM_CustomNonbondedForce):
    """
    A damped-smoothed version of the Lennard-Jones/Coulomb potential.

    .. math::
        & V(r)=\\left\\{
            4\\epsilon\\left[
                \\left(\\frac{\\sigma}{r}\\right)^{12}-\\left(\\frac{\\sigma}{r}\\right)^6
            \\right]+\\frac{q_1 q_2}{4\\pi\\epsilon_0}\\frac{\\mathrm{erfc}(r)}{r}
        \\right\\}S(r) \\\\
        & \\sigma=\\frac{\\sigma_1+\\sigma_2}{2} \\\\
        & \\epsilon=\\sqrt{\\epsilon_1\\epsilon_2} \\\\
        & S(r)=[1+\\theta(r-r_\\mathrm{switch})u^3(15u-6u^2-10)] \\\\
        & u=\\frac{r^n-r_\\mathrm{switch}^n}{r_\\mathrm{cut}^n-r_\\mathrm{switch}^n}

    .. warning::
        Long-range dispersion correction is not employed.

    In the equations above, :math:`\\theta(x)` is the Heaviside step function. Note that the
    switching function employed here, with `u` being a quadratic function of `r`, is slightly
    different from the one normally used in OpenMM, in which `u` is a linear function of `r`.

    Parameters
    ----------
        alpha : Number or unit.Quantity
            The Coulomb damping parameter (in inverse distance unit).
        cutoff_distance : Number or unit.Quantity
            The distance at which the nonbonded interaction vanishes.
        switch_distance : Number or unit.Quantity
            The distance at which the switching function begins to smooth the approach of the
            nonbonded interaction towards zero.
        degree : int, optional, default=1
            The degree `n` in the definition of the switching variable `u` (see above).

    """
    def __init__(self, alpha, cutoff_distance, switch_distance, degree=1):
        if switch_distance/switch_distance.unit < 0.0 or switch_distance >= cutoff_distance:
            raise atomsmm.utils.InputError('Switching distance must satisfy 0 <= r_switch < r_cutoff')
        LennardJones = '4*epsilon*((sigma/r)^12 - (sigma/r)^6)'
        Coulomb = 'Kc*chargeprod/r'
        if degree == 1:
            energy = '{} + erfc(alpha*r)*{}'.format(LennardJones, Coulomb)
        else:
            energy = 'S*({} + erfc(alpha*r)*{});'.format(LennardJones, Coulomb)
            energy += 'S = 1 + step(r - rswitch)*u^3*(15*u - 6*u^2 - 10);'
            energy += 'u = (r^d - rswitch^d)/(rcut^d - rswitch^d); d={}'.format(degree)
        super().__init__(
            energy=energy,
            cutoff_distance=cutoff_distance,
            use_switching_function=(degree == 1),
            switch_distance=(switch_distance if degree == 1 else None),
            use_dispersion_correction=False,
            Kc=138.935456*unit.kilojoules_per_mole/unit.nanometer,
            alpha=alpha,
            rswitch=switch_distance,
            rcut=cutoff_distance,
        )


def nearForceExpressions(cutoff_distance, switch_distance, adjustment):
    expressions = []
    if adjustment is None:
        expressions.append('energy=S*(4*epsilon*((sigma/r)^12-(sigma/r)^6) + Kc*chargeprod/r)')
        expressions.append('S = 1 + step(r - rs0)*u^3*(15*u - 6*u^2 - 10)')
    elif adjustment == 'shift':
        LJ = '4*epsilon*((sigma/r)^12-(sigma/r)^6-((sigma/rc0)^12-(sigma/rc0)^6))'
        Coulomb = 'Kc*chargeprod*(1/r-1/rc0)'
        expressions.append('S*({}+{})'.format(LJ, Coulomb))
        expressions.append('S = 1 + step(r - rs0)*u^3*(15*u - 6*u^2 - 10)')
    elif adjustment == 'force-switch':
        potential = '4*epsilon*(f12*(sigma/r)^12-f6*(sigma/r)^6) + Kc*chargeprod*f1/r'
        shift = '4*epsilon*(f12c*(sigma/rc0)^12-f6c*(sigma/rc0)^6) + Kc*chargeprod*f1c/rc0'
        factors = dict(f12='(6*b^2-21*b+28)*(b^3*(R^12-1)-12*b^2*u-66*b*u^2-220*u^3)/462+45*(7-2*b)*u^4/14-72*u^5/7',
                       f6='(6*b^2-3*b+1)*(b^3*(R^6-1)-6*b^2*u-15*b*u^2-20*u^3)+45*(1-2*b)*u^4-36*u^5',
                       f1='5*(b+1)^2*(6*b^3*R*log(R)-6*b^2*u-3*b*u^2+u^3)+u^4*(3*u-5*b-10)/2')
        expressions.append('{}-({})'.format(potential, shift))
        for factor, func in factors.items():
            expressions.append('{}=1+step(r-rs0)*({})'.format(factor, func))
        expressions.append('R=u/b+1')  # R=r/rs0
        b = switch_distance/(cutoff_distance-switch_distance)
        expressions.append('b={}'.format(b))
        expressions.append('f12c={}'.format((1+b)**3*(b**6+3*b**5+(30/7)*b**4+(25/7)*b**3+(25/14)*b**2+(1/2)*b+2/33)/b**9))
        expressions.append('f6c={}'.format((1+b)**3/b**3))
        expressions.append('f1c={}'.format((30*(1+b))*(b**2*(1+b)**2*math.log(1/b+1)-b**3-(3/2)*b**2-(1/3)*b+1/12)))
    else:
        raise atomsmm.utils.InputError('unknown adjustment option')
    expressions.append('u=(r-rs0)/(rc0-rs0)')
    expressions.append('rs0={}'.format(switch_distance.value_in_unit(unit.nanometers)))
    expressions.append('rc0={}'.format(cutoff_distance.value_in_unit(unit.nanometers)))
    expressions.append('Kc=138.935456')
    return expressions


class NearForce(object):
    def _globalParams(self, cutoff_distance, switch_distance):
        return {'Kc': 138.935456*unit.kilojoules_per_mole/unit.nanometer,
                'rc0': cutoff_distance,
                'rs0': switch_distance}

    def _expressions(self, cutoff_distance, switch_distance, adjustment):
        expressions = []
        if adjustment is None:
            expressions.append('S*(4*epsilon*((sigma/r)^12-(sigma/r)^6) + Kc*chargeprod/r)')
            expressions.append('S = 1 + step(r - rs0)*u^3*(15*u - 6*u^2 - 10)')
        elif adjustment == 'shift':
            LJ = '4*epsilon*((sigma/r)^12-(sigma/r)^6-((sigma/rc0)^12-(sigma/rc0)^6))'
            Coulomb = 'Kc*chargeprod*(1/r-1/rc0)'
            expressions.append('S*({}+{})'.format(LJ, Coulomb))
            expressions.append('S = 1 + step(r - rs0)*u^3*(15*u - 6*u^2 - 10)')
        elif adjustment == 'force-switch':
            potential = '4*epsilon*(f12*(sigma/r)^12-f6*(sigma/r)^6) + Kc*chargeprod*f1/r'
            shift = '4*epsilon*(f12c*(sigma/rc0)^12-f6c*(sigma/rc0)^6) + Kc*chargeprod*f1c/rc0'
            factors = dict(f12='(6*b^2-21*b+28)*(b^3*(R^12-1)-12*b^2*u-66*b*u^2-220*u^3)/462+45*(7-2*b)*u^4/14-72*u^5/7',
                           f6='(6*b^2-3*b+1)*(b^3*(R^6-1)-6*b^2*u-15*b*u^2-20*u^3)+45*(1-2*b)*u^4-36*u^5',
                           f1='5*(b+1)^2*(6*b^3*R*log(R)-6*b^2*u-3*b*u^2+u^3)+u^4*(3*u-5*b-10)/2')
            expressions.append('{}-({})'.format(potential, shift))
            for factor, func in factors.items():
                expressions.append('{}=1+step(r-rs0)*({})'.format(factor, func))
            expressions.append('R=u/b+1')  # R=r/rs0
            b = switch_distance/(cutoff_distance-switch_distance)
            expressions.append('b={}'.format(b))
            expressions.append('f12c={}'.format((1+b)**3*(b**6+3*b**5+(30/7)*b**4+(25/7)*b**3+(25/14)*b**2+(1/2)*b+2/33)/b**9))
            expressions.append('f6c={}'.format((1+b)**3/b**3))
            expressions.append('f1c={}'.format((30*(1+b))*(b**2*(1+b)**2*math.log(1/b+1)-b**3-(3/2)*b**2-(1/3)*b+1/12)))
        else:
            raise atomsmm.utils.InputError('unknown adjustment option')
        expressions.append('u=(r-rs0)/(rc0-rs0)')
        return expressions


class NearNonbondedForce(_AtomsMM_CustomNonbondedForce, NearForce):
    """
    This is a smoothed version of the Lennard-Jones + Coulomb potential

    .. math::
        V_\\mathrm{LJC}(r)=4\\epsilon\\left[
                \\left(\\frac{\\sigma}{r}\\right)^{12}-\\left(\\frac{\\sigma}{r}\\right)^6
            \\right]+\\frac{1}{4\\pi\\epsilon_0}\\frac{q_1 q_2}{r}.

    The smoothing is accomplished by application of a 5th-order spline function :math:`S(u(r))`, which
    varies softly from 1 down to 0 along the range :math:`r_\\mathrm{switch} \\leq r \\leq r_\\mathrm{cut}`.
    Such function is

    .. math::
        S(u)=1+u^3(15u-6u^2-10),

    where

    .. math::
        u(r)=\\begin{cases}
                 0 & r < r_\\mathrm{switch} \\\\
                 \\dfrac{r-r_\\mathrm{switch}}{r_\\mathrm{cut}-r_\\mathrm{switch}} &
                     r_\\mathrm{switch} \\leq r \\leq r_\\mathrm{cut} \\\\
                 1 & r > r_\\mathrm{cut}
             \\end{cases}.

    Such type of smoothing is essential for application in multiple time-scale integration using
    the RESPA-2 scheme described in Refs. :cite:`Zhou_2001`, :cite:`Morrone_2010`, and
    :cite:`Leimkuhler_2013`.

    Three distinc versions are available:

    1. Applying the switch directly to the potential:

    .. math::
        V(r)=S(u(r))V_\\mathrm{LJC}(r).

    2. Applying the switch to a shifted version of the potential:

    .. math::
        V(r)=S(u(r))\\left[V_\\mathrm{LJC}(r)-V_\\mathrm{LJC}(r_\\mathrm{cut})\\right]

    3. Applying the switch to the force that results from the potential:

    .. math::
        & V(r)=V^\\ast_\\mathrm{LJC}(r)-V^\\ast_\\mathrm{LJC}(r_\\mathrm{cut}) \\\\
        & V^\\ast_\\mathrm{LJC}(r)=\\left\\{
            4\\epsilon\\left[f_{12}(u(r))\\left(\\frac{\\sigma}{r}\\right)^{12}-f_6(u(r))\\left(\\frac{\\sigma}{r}\\right)^6\\right]
            + \\frac{f_1(u(r))}{4\\pi\\epsilon_0}\\frac{q_1 q_2}{r}
        \\right\\}

    where :math:`f_n(u)` is the solution of the 1st order differential equation

    .. math::
        & f_n-\\frac{u+b}{n}\\frac{df_n}{du}=S(u) \\\\
        & f_n(0)=1 \\\\
        & b=\\frac{r_\\mathrm{switch}}{r_\\mathrm{cut}-r_\\mathrm{switch}}

    As a consequence of this modification, :math:`V^\\prime(r)=S(u(r))V^\\prime_\\mathrm{LJC}(r)`.

    .. note::
        In all cases, the Lorentz-Berthelot mixing rule is applied for unlike-atom interactions.

    Parameters
    ----------
        cutoff_distance : unit.Quantity
            The distance at which the nonbonded interaction vanishes.
        switch_distance : unit.Quantity
            The distance at which the switching function begins to smooth the approach of the
            nonbonded interaction towards zero.
        adjustment : str, optional, default=None
            A keyword for modifying the potential energy function. If it is `None`, then the
            switching function is applied directly to the original potential. Other options are
            `'shift'` and `'force-switch'`. If it is `'shift'`, then the switching function is
            applied to a potential that is already null at the cutoff due to a previous shift.
            If it is `'force-switch'`, then the potential is modified so that the switching
            function is applied to the forces rather than the potential energy.
        subtract : bool, optional, default=False
            Whether to substract (rather than add) the force.
        actual_cutoff : unit.Quantity, optional, default=None
            The cutoff that will actually be used by OpenMM. This is often required for
            compatibility with other forces in the same force group. If it is `None`, then the
            passed `cutoff_distance` (see above) will be used.

    """
    def __init__(self, cutoff_distance, switch_distance, adjustment=None, subtract=False,
                 actual_cutoff=None):
        globalParams = self._globalParams(cutoff_distance, switch_distance)
        expressions = self._expressions(cutoff_distance, switch_distance, adjustment)
        rcut = cutoff_distance if actual_cutoff is None else actual_cutoff
        if actual_cutoff is not None:
            expressions[0] = 'step(rc0-r)*({})'.format(expressions[0])
        if subtract:
            expressions[0] = '-({})'.format(expressions[0])
        super().__init__(
            energy='; '.join(expressions),
            cutoff_distance=rcut,
            use_switching_function=False,
            use_dispersion_correction=False,
            **globalParams,
        )


class NearExceptionForce(_AtomsMM_CustomBondForce, NearForce):
    def __init__(self, cutoff_distance, switch_distance, adjustment=None, subtract=False):
        globalParams = self._globalParams(cutoff_distance, switch_distance)
        expressions = self._expressions(cutoff_distance, switch_distance, adjustment)
        expressions[0] = 'step(rc0-r)*({})'.format(expressions[0])
        if subtract:
            expressions[0] = '-{}'.format(expressions[0])
        super().__init__('; '.join(expressions), **globalParams)


class FarNonbondedForce(_AtomsMM_CompoundForce):
    """
    The complement of NearNonbondedForce and NonbondedExceptionsForce classes in order to form a
    complete OpenMM NonbondedForce.

    .. note::
        Except for the shifting, this model is the 'far' part of the RESPA2 scheme of
        Refs. :cite:`Zhou_2001` and :cite:`Morrone_2010`, with the switching function applied to
        the potential rather than to the force.

    Parameters
    ----------
        preceding : :class:`NearNonbondedForce`
            The NearNonbondedForce object with which this Force is supposed to match.
        cutoff_distance : Number or unit.Quantity
            The distance at which the nonbonded interaction vanishes.
        switch_distance : Number or unit.Quantity, optional, default=None
            The distance at which the switching function begins to smooth the approach of the
            nonbonded interaction towards zero. If this is None, then no switching will be done
            prior to the potential cutoff.
        nonbondedMethod : openmm.NonbondedForce.Method, optional, default=PME
            The method to use for nonbonded interactions. Allowed values are NoCutoff,
            CutoffNonPeriodic, CutoffPeriodic, Ewald, PME, or LJPME.
        ewaldErrorTolerance : Number, optional, default=1E-5
            The error tolerance for Ewald summation.

    """
    def __init__(self, preceding, cutoff_distance, switch_distance=None):
        if not isinstance(preceding, NearNonbondedForce):
            raise atomsmm.utils.InputError('argument \'preceding\' must be of class NearNonbondedForce')
        potential = preceding.getEnergyFunction().split(';')
        potential[0] = '-step(rc0-r)*({})'.format(potential[0])
        expression = ';'.join(potential)
        discount = _AtomsMM_CustomNonbondedForce(
            energy=expression,
            cutoff_distance=cutoff_distance,
            use_switching_function=False,
            use_dispersion_correction=False,
            **preceding.getGlobalParameters(),
        )
        total = _AtomsMM_NonbondedForce(cutoff_distance, switch_distance)
        super().__init__([total, discount])


class SoftcoreLennardJonesForce(_AtomsMM_CustomNonbondedForce):
    """
    A softened version of the Lennard-Jones potential.

    .. math::
        & V(r)=4\\lambda\\epsilon\\left(\\frac{1}{s^2} - \\frac{1}{s}\\right) \\\\
        & s = \\left(\\frac{r}{\\sigma}\\right)^6 + \\frac{1}{2}(1-\\lambda) \\\\
        & \\sigma=\\frac{\\sigma_1+\\sigma_2}{2} \\\\
        & \\epsilon=\\sqrt{\\epsilon_1\\epsilon_2}

    Parameters
    ----------
        cutoff_distance : Number or unit.Quantity
            The distance at which the nonbonded interaction vanishes.
        switch_distance : Number or unit.Quantity
            The distance at which the switching function begins to smooth the approach of the
            nonbonded interaction towards zero.

    """
    def __init__(self, cutoff_distance=None, use_switching_function=None,
                 switch_distance=None, use_dispersion_correction=None, parameter='lambda'):
        globalParams = {parameter: 1.0}
        potential = '4*{}*epsilon*x*(x-1);'.format(parameter)
        potential += 'x = 1/((r/sigma)^6 + 0.5*(1-{}))'.format(parameter)
        super().__init__(
            energy=potential,
            cutoff_distance=cutoff_distance,
            use_switching_function=use_switching_function,
            switch_distance=switch_distance,
            use_dispersion_correction=use_dispersion_correction,
            **globalParams,
        )


class SoftcoreForce(_AtomsMM_CustomNonbondedForce):
    """
    A softened version of the Lennard-Jones+Coulomb potential.

    .. math::
        & V(r) = V_\\mathrm{vdw}(r) + V_\\mathrm{coul}(r)
        & V_\\mathrm{vdw}(r)=4\\lambda_\\mathrm{vdw}\\epsilon\\left(\\frac{1}{s^2} - \\frac{1}{s}\\right) \\\\
        & s = \\left(\\frac{r}{\\sigma}\\right)^6 + \\frac{1}{2}(1-\\lambda_\\mathrm{vdw}) \\\\
        & \\sigma=\\frac{\\sigma_1+\\sigma_2}{2} \\\\
        & \\epsilon=\\sqrt{\\epsilon_1\\epsilon_2}
        & V_\\mathrm{coul}(r)=\\lambda_\\mathrm{coul}\\frac{q_1 q_2}{4\\pi\\epsilon_0}\\frac{1}{r}

    Parameters
    ----------
        cutoff_distance : Number or unit.Quantity
            The distance at which the nonbonded interaction vanishes.
        switch_distance : Number or unit.Quantity
            The distance at which the switching function begins to smooth the approach of the
            nonbonded interaction towards zero.

    """
    def __init__(self, cutoff_distance, switch_distance=None):
        globalParams = {'Kc': 138.935456*unit.kilojoules_per_mole/unit.nanometer,
                        'lambda_vdw': 1.0, 'lambda_coul': 1.0}
        potential = '4*lambda_vdw*epsilon*(1-x)/x^2 + Kc*lambda_coul*chargeprod/r;'
        potential += 'x = (r/sigma)^6 + 0.5*(1-lambda_vdw)'
        super().__init__(
            energy=potential,
            cutoff_distance=cutoff_distance,
            use_switching_function=True,
            switch_distance=switch_distance,
            **globalParams,
        )
