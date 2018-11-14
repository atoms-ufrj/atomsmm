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

from atomsmm.utils import Coulomb
from atomsmm.utils import InputError
from atomsmm.utils import LennardJones
from atomsmm.utils import LorentzBerthelot


class _Proxy(object):
    def __init__(self, force, method):
        self.force = force
        self.method = method

    def __call__(self, *args, **kwargs):
        for instance in self.force:
            if hasattr(instance, self.method):
                getattr(instance, self.method)(*args, **kwargs)
        return self.force


class AtomsMMForce:
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
        Returns a proxy object which, in turn, will apply the `method`, with the same arguments of
        the original call, to all OpenMM forces which are stored in `self` and have `method` as an
        attribute. In the end, the proxy will return that AtomsMM force for the purpose of chaining.
        """
        return _Proxy(self, method)

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
        exceptions = _AtomsMMCustomBondForce()
        exceptions.setForceGroup(self.getForceGroup())
        self.forces.append(exceptions)
        return self


class _AtomsMMNonbondedForce(openmm.NonbondedForce):
    """
    An extension of OpenMM's NonbondedForce_ class, but without non-exclusion exceptions. These must
    be handled separately using a :class:`_AtomsMMCustomBondForce` object.

    .. warning::
        When the :func:`~_AtomsMMNonbondedForce.importFrom` is used, all exceptions of the OpenMM
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
            :class:`_AtomsMMNonbondedForce`
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


class _AtomsMMCustomNonbondedForce(openmm.CustomNonbondedForce):
    """
    An extension of OpenMM's CustomNonbondedForce_ class.

    Parameters
    ----------
        energy : str
            An algebraic expression giving the interaction energy between two particles as a
            function of their distance `r`, as well as any per-particle and global parameters.
        cutoff_distance : Number or unit.Quantity
            The cutoff distance being used for nonbonded interactions.
        switch_distance :  Number or unit.Quantity, optional, default=None
            The distance at which the switching function begins to reduce the interaction. If this
            is None, then no switching will be done.
        parameters : list(str), optional, default=["charge", "sigma", "epsilon"]
            A list of nonbonded force parameters that depend on the individual particles.
        **kwargs
            Keyword arguments defining names and values of global nonbonded force parameters.

    """
    def __init__(self, energy, cutoff_distance, switch_distance=None, usesCharges=True, **globalParams):
        super().__init__(energy)
        self.usesCharges = usesCharges
        if self.usesCharges:
            self.addPerParticleParameter("charge")
        self.addPerParticleParameter("sigma")
        self.addPerParticleParameter("epsilon")
        for (name, value) in globalParams.items():
            self.addGlobalParameter(name, value)
        self.setCutoffDistance(cutoff_distance)
        if switch_distance is None:
            self.setUseSwitchingFunction(False)
        else:
            self.setUseSwitchingFunction(True)
            self.setSwitchingDistance(switch_distance)

    def importFrom(self, force):
        """
        Import all particles and exceptions from a passed OpenMM NonbondedForce_ object while
        transforming all non-exclusion exceptions into exclusion ones.

        Parameters
        ----------
            force : openmm.NonbondedForce
                The force from which the particles and exclusions will be imported.

        Returns
        -------
            :class:`_AtomsMMCustomNonbondedForce`
                The object is returned for chaining purposes.

        """
        for index in range(force.getNumParticles()):
            charge, sigma, epsilon = force.getParticleParameters(index)
            if self.usesCharges:
                self.addParticle([charge, sigma, epsilon])
            else:
                self.addParticle([sigma, epsilon])
        for index in range(force.getNumExceptions()):
            i, j, chargeprod, sigma, epsilon = force.getExceptionParameters(index)
            self.addExclusion(i, j)

        builtin = openmm.NonbondedForce
        custom = openmm.CustomNonbondedForce
        mapping = {builtin.CutoffNonPeriodic: custom.CutoffNonPeriodic,
                   builtin.CutoffPeriodic: custom.CutoffPeriodic,
                   builtin.Ewald: custom.CutoffPeriodic,
                   builtin.NoCutoff: custom.NoCutoff,
                   builtin.PME: custom.CutoffPeriodic}
        self.setNonbondedMethod(mapping[force.getNonbondedMethod()])
        if not self.usesCharges:
            self.setUseLongRangeCorrection(force.getUseDispersionCorrection())
        return self


class _AtomsMMCustomBondForce(openmm.CustomBondForce):
    """
    An extension of OpenMM's CustomBondForce_ class used to handle NonbondedForce exceptions.

    """
    def __init__(self):
        super().__init__("4*epsilon*x*(x-1) + Kc*chargeprod/r; x=(sigma/r)^6")
        self.addGlobalParameter("Kc", 138.935456*unit.kilojoules_per_mole/unit.nanometer)
        self.addPerBondParameter("chargeprod")
        self.addPerBondParameter("sigma")
        self.addPerBondParameter("epsilon")

    def importFrom(self, force):
        """
        Import all non-exclusion exceptions from the a passed OpenMM NonbondedForce_ object.

        Parameters
        ----------
            force : openmm.NonbondedForce
                The force from which the exceptions will be imported.

        Returns
        -------
            :class:`Force`
                The object is returned for chaining purposes.

        """
        for index in range(force.getNumExceptions()):
            i, j, chargeprod, sigma, epsilon = force.getExceptionParameters(index)
            if chargeprod/chargeprod.unit != 0.0 or epsilon/epsilon.unit != 0.0:
                self.addBond(i, j, [chargeprod, sigma, epsilon])
        return self


class DampedSmoothedForce(AtomsMMForce):
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
            raise InputError("Switching distance must satisfy 0 <= r_switch < r_cutoff")
        if degree == 1:
            energy = "{} + erfc(alpha*r)*{};".format(LennardJones("r"), Coulomb("r"))
        else:
            energy = "S*({} + erfc(alpha*r)*{});".format(LennardJones("r"), Coulomb("r"))
            energy += "S = 1 + step(r - rswitch)*u^3*(15*u - 6*u^2 - 10);"
            energy += "u = (r^d - rswitch^d)/(rcut^d - rswitch^d); d={};".format(degree)
        energy += LorentzBerthelot()
        force = _AtomsMMCustomNonbondedForce(energy, cutoff_distance,
                                             switch_distance if degree == 1 else None,
                                             Kc=138.935456*unit.kilojoules_per_mole/unit.nanometer,
                                             alpha=alpha, rswitch=switch_distance, rcut=cutoff_distance)
        super().__init__(force)


class NonbondedExceptionsForce(AtomsMMForce):
    """
    A special class designed to compute only the exceptions of an OpenMM NonbondedForce object.

    """
    def __init__(self):
        super().__init__(_AtomsMMCustomBondForce())


class NearNonbondedForce(AtomsMMForce):
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
        cutoff_distance : Number or unit.Quantity
            The distance at which the nonbonded interaction vanishes.
        switch_distance : Number or unit.Quantity
            The distance at which the switching function begins to smooth the approach of the
            nonbonded interaction towards zero.
        shifted : Bool, optional, default=True
            If True, a potential shift is done for both the Lennard-Jones and the Coulomb term
            prior to the potential smoothing.
        adjustment : str, optional, default=None
            A keyword for modifying the potential energy function. If it is `None`, then the
            switching function is applied directly to the original potential. Other options are
            `"shift"` and `"force-switch"`. If it is `"shift"`, then the switching function is
            applied to a potential that is already null at the cutoff due to a previous shift.
            If it is `"force-switch"`, then the potential is modified so that the switching
            function is applied to the forces rather than the potential energy.

    """
    def __init__(self, cutoff_distance, switch_distance, adjustment=None):
        self.globalParams = {"Kc": 138.935456*unit.kilojoules_per_mole/unit.nanometer,
                             "rc0": cutoff_distance,
                             "rs0": switch_distance}
        if adjustment is None:
            expression = "S*(4*epsilon*((sigma/r)^12-(sigma/r)^6) + Kc*chargeprod/r);"
            expression += "S = 1 + step(r - rs0)*u^3*(15*u - 6*u^2 - 10);"
        elif adjustment == "shift":
            expression = "S*(4*epsilon*((sigma/r)^12-(sigma/r)^6-((sigma/rc0)^12-(sigma/rc0)^6)) + Kc*chargeprod*(1/r-1/rc0));"
            expression += "S = 1 + step(r - rs0)*u^3*(15*u - 6*u^2 - 10);"
        elif adjustment == "force-switch":
            potential = "4*epsilon*(f12*(sigma/r)^12-f6*(sigma/r)^6) + Kc*chargeprod*f1/r"
            shift = "4*epsilon*(f12c*(sigma/rc0)^12-f6c*(sigma/rc0)^6) + Kc*chargeprod*f1c/rc0"
            expression = "{}-({});".format(potential, shift)
            expression += "f12=1+step(r-rs0)*((6*b^2-21*b+28)*(b^3*(R^12-1)-12*b^2*u-66*b*u^2-220*u^3)/462+45*(7-2*b)*u^4/14-72*u^5/7);"
            expression += "f6=1+step(r-rs0)*((6*b^2-3*b+1)*(b^3*(R^6-1)-6*b^2*u-15*b*u^2-20*u^3)+45*(1-2*b)*u^4-36*u^5);"
            expression += "f1=1+step(r-rs0)*(5*(b+1)^2*(6*b^3*R*log(R)-6*b^2*u-3*b*u^2+u^3)+u^4*(3*u-5*b-10)/2);"
            expression += "R=u/b+1;"  # R=r/rs0
            b = switch_distance/(cutoff_distance-switch_distance)
            expression += "b={};".format(b)
            expression += "f12c={};".format((1+b)**3*(b**6+3*b**5+(30/7)*b**4+(25/7)*b**3+(25/14)*b**2+(1/2)*b+2/33)/b**9)
            expression += "f6c={};".format((1+b)**3/b**3)
            expression += "f1c={};".format((30*(1+b))*(b**2*(1+b)**2*math.log(1/b+1)-b**3-(3/2)*b**2-(1/3)*b+1/12))
        else:
            raise InputError("unknown adjustment option")
        expression += "u=(r-rs0)/(rc0-rs0);"
        expression += LorentzBerthelot()
        force = _AtomsMMCustomNonbondedForce(expression, cutoff_distance, None, **self.globalParams)
        super().__init__(force)
        self.expression = expression


class FarNonbondedForce(AtomsMMForce):
    """
    The complement of NearNonbondedForce and NonbondedExceptionsForce classes in order to form a
    complete OpenMM NonbondedForce.

    .. note::
        Except for the shifting, this model is the "far" part of the RESPA2 scheme of
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
            raise InputError("argument 'preceding' must be of class NearNonbondedForce")
        potential = preceding.expression.split(";")
        potential[0] = "-step(rc0-r)*({})".format(potential[0])
        expression = ";".join(potential)
        discount = _AtomsMMCustomNonbondedForce(expression, cutoff_distance, None, **preceding.globalParams)
        total = _AtomsMMNonbondedForce(cutoff_distance, switch_distance)
        super().__init__([total, discount])


class SoftcoreLennardJonesForce(AtomsMMForce):
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
    def __init__(self, cutoff_distance, switch_distance=None):
        globalParams = {"lambda": 1.0}
        potential = "4*lambda*epsilon*(1-x)/x^2; x = (r/sigma)^6 + 0.5*(1-lambda);" + LorentzBerthelot()
        force = _AtomsMMCustomNonbondedForce(potential, cutoff_distance, switch_distance, **globalParams)
        super().__init__(force)
