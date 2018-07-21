"""
.. module:: forces
   :platform: Unix, Windows
   :synopsis: a module for defining the basis class :class:`Force` and derived classes.

.. moduleauthor:: Charlles R. A. Abreu <abreu@eq.ufrj.br>

"""

from simtk import openmm
from simtk import unit

from atomsmm.utils import InputError


class Force:
    """
    The basis class of every AtomsMM Force object, which is a list of OpenMM Force_ objects treated
    as a single force.

    .. _Force: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.Force.html

    Parameters
    ----------
        forces : list(openmm.Force), optional, default=[]
            A list of OpenMM Force objects.

    """
    def __init__(self, forces=[]):
        self.forces = forces
        self.setForceGroup(0)

    def setForceGroup(self, group):
        """
        Set the force group to which this :class:`Force` object belongs.

        Parameters
        ----------
            group : int
                The group index. Legal values are between 0 and 31 (inclusive).

        Returns
        -------
            :class:`Force`
                Although the operation is done inline, the modified Force is returned for chaining
                purposes.

        """
        for force in self.forces:
            force.setForceGroup(group)
        return self

    def getForceGroup(self):
        """
        Get the force group to which this :class:`Force` object belongs.

        Returns
        -------
            int
                The group index, whose value is between 0 and 31 (inclusive).

        """
        return self.forces[0].getForceGroup() if self.forces else 0

    def includeExceptions(self):
        """
        Incorporate non-exclusion exceptions when importing parameters via method
        :func:`~Force.importFrom`.

        Returns
        -------
            :class:`Force`
                The object is returned for chaining purposes.

        """
        exceptions = _ExceptionNonbondedForce()
        exceptions.setForceGroup(self.getForceGroup())
        self.forces.append(exceptions)
        return self

    def addTo(self, system):
        """
        Add the :class:`Force` object to an OpenMM System_ object.

        .. _System: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.System.html

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

    def importFrom(self, nbForce):
        """
        Import parameters from a provided OpenMM NonbondedForce_ object.

        .. _NonbondedForce: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.NonbondedForce.html

        Parameters
        ----------
            nbForce : openmm.NonbondedForce
                The force from which the parameters will be imported.

        Returns
        -------
            :class:`Force`
                The object is returned for chaining purposes.

        """
        for force in self.forces:
            force.importFrom(nbForce)
        return self


class _NonbondedForce(openmm.NonbondedForce):
    """
    An extension of OpenMM's NonbondedForce_ class, but with exclusion exceptions only. By default,
    long-range dispersion correction is employed and the method used for long-range electrostatic
    interactions is `PME`.

    ..note:
        Non-exclusion exceptions must be handled separately using :class:`_ExceptionNonbondedForce`
        objects.

    .. _NonbondedForce: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.NonbondedForce.html

    Parameters
    ----------
        cutoff_distance : Number or unit.Quantity
            The cutoff distance being used for nonbonded interactions.
        switch_distance :  Number or unit.Quantity, optional, default=None
            The distance at which the switching function begins to reduce the interaction. If this
            is None, then no switching will be done.

    """
    def __init__(self, cutoff_distance, switch_distance=None):
        super(_NonbondedForce, self).__init__()
        self.setNonbondedMethod(openmm.NonbondedForce.PME)
        self.setCutoffDistance(cutoff_distance)
        self.setUseDispersionCorrection(True)
        if switch_distance is None:
            self.setUseSwitchingFunction(False)
        else:
            self.setUseSwitchingFunction(True)
            self.setSwitchingDistance(switch_distance)

    def importFrom(self, force):
        """
        Import all particles and exceptions from the a passed OpenMM NonbondedForce_ object and
        turn all non-exclusion exceptions into exclusion ones.

        .. _NonbondedForce: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.NonbondedForce.html

        Parameters
        ----------
            force : openmm.NonbondedForce
                The force from which the particles and exclusions will be imported.

        Returns
        -------
            :class:`Force`
                The object is returned for chaining purposes.

        """
        for index in range(force.getNumParticles()):
            self.addParticle(*force.getParticleParameters(index))
        for index in range(force.getNumExceptions()):
            i, j, chargeProd, sigma, epsilon = force.getExceptionParameters(index)
            self.addException(i, j, 0.0*chargeProd, sigma, 0.0*epsilon)
        return self


class _CustomNonbondedForce(openmm.CustomNonbondedForce):
    """
    An extension of OpenMM's CustomNonbondedForce_ class.

    .. _CustomNonbondedForce: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.CustomNonbondedForce.html

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
    def __init__(self, energy, cutoff_distance, switch_distance=None,
                 parameters=["charge", "sigma", "epsilon"], **kwargs):
        super(_CustomNonbondedForce, self).__init__(energy)
        for name in parameters:
            self.addPerParticleParameter(name)
        for (name, value) in kwargs.items():
            self.addGlobalParameter(name, value)
        self.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
        self.setCutoffDistance(cutoff_distance)
        self.setUseLongRangeCorrection(False)
        if switch_distance is None:
            self.setUseSwitchingFunction(False)
        else:
            self.setUseSwitchingFunction(True)
            self.setSwitchingDistance(switch_distance)

    def importFrom(self, force):
        """
        Import all particles and exclusions from the a passed OpenMM NonbondedForce_ object.

        .. _NonbondedForce: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.NonbondedForce.html

        Parameters
        ----------
            force : openmm.NonbondedForce
                The force from which the particles and exclusions will be imported.

        Returns
        -------
            :class:`Force`
                The object is returned for chaining purposes.

        """
        for index in range(force.getNumParticles()):
            self.addParticle(force.getParticleParameters(index))
        for index in range(force.getNumExceptions()):
            i, j, chargeProd, sigma, epsilon = force.getExceptionParameters(index)
            if chargeProd/chargeProd.unit == 0.0 and epsilon/epsilon.unit == 0.0:
                self.addExclusion(i, j)
        return self


class _ExceptionNonbondedForce(openmm.CustomBondForce):
    """
    A special class for handling OpenMM NonbondedForce_ exceptions.

    .. _NonbondedForce: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.NonbondedForce.html

    """
    def __init__(self):
        energy = "4*epsilon*((sigma/r)^12-(sigma/r)^6) + Kc*chargeprod/r;"
        super(_ExceptionNonbondedForce, self).__init__(energy)
        self.addGlobalParameter("Kc", 138.935456*unit.kilojoules/unit.nanometer)
        self.addPerBondParameter("chargeprod")
        self.addPerBondParameter("sigma")
        self.addPerBondParameter("epsilon")

    def importFrom(self, force):
        """
        Import all non-exclusion exceptions from the a passed OpenMM NonbondedForce_ object.

        .. _NonbondedForce: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.NonbondedForce.html

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
            [i, j, chargeProd, sigma, epsilon] = force.getExceptionParameters(index)
            if (chargeProd/chargeProd.unit != 0.0) or (epsilon/epsilon.unit != 0.0):
                self.addBond(i, j, [chargeProd, sigma, epsilon])
        return self


class DampedSmoothedForce(Force):
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
            The damping parameter (in inverse distance unit).
        rswitch : Number or unit.Quantity
            The distance marking the start of the switching range.
        rcut : Number or unit.Quantity
            The potential cut-off distance.
        degree : int, optional, default=1
            The degree `n` in the definition of the switching variable `u` (see above).

    """
    def __init__(self, alpha, rswitch, rcut, degree=1):
        energy = "(4*epsilon*((sigma/r)^12-(sigma/r)^6) + Kc*charge1*charge2*erfc(alpha*r)/r)*f;"
        if degree == 1:
            energy += "f = 1;"
        else:
            energy += "f = 1 + step(r - rswitch)*u^3*(15*u - 6*u^2 - 10);"
            energy += "u = (r^%d - rswitch^%d)/(rcut^%d - rswitch^%d);" % ((degree,)*4)
        energy += "sigma = 0.5*(sigma1+sigma2);"
        energy += "epsilon = sqrt(epsilon1*epsilon2);"
        force = _CustomNonbondedForce(energy, rcut, rswitch if degree == 1 else None,
                                      Kc=138.935456*unit.kilojoules/unit.nanometer,
                                      alpha=alpha, rswitch=rswitch, rcut=rcut)
        super(DampedSmoothedForce, self).__init__([force])


class InnerRespaForce(Force):
    """
    A smoothed version of the (optionally) shifted Lennard-Jones+Coulomb potential.

    .. math::
        & V(r)=\\left[U(r)-\\delta_\\mathrm{shift}U(r_\\mathrm{cut})\\right]S(r) \\\\
        & U(r)=4\\epsilon\\left[
                \\left(\\frac{\\sigma}{r}\\right)^{12}-\\left(\\frac{\\sigma}{r}\\right)^6
            \\right]+\\frac{1}{4\\pi\\epsilon_0}\\frac{q_1 q_2}{r} \\\\
        & \\sigma=\\frac{\\sigma_1+\\sigma_2}{2} \\\\
        & \\epsilon=\\sqrt{\\epsilon_1\\epsilon_2} \\\\
        & S(r)=\\theta(r_\\mathrm{cut}-r)[1+\\theta(r-r_\\mathrm{switch})u^3(15u-6u^2-10)] \\\\
        & u=\\frac{r-r_\\mathrm{switch}}{r_\\mathrm{cut}-r_\\mathrm{switch}}

    In the equations above, :math:`\\theta(x)` is the Heaviside step function. The constant
    :math:`\\delta_\\mathrm{shift}` is the numerical value (that is, 1 or 0) of the optional
    boolean argument `shift`.

    Parameters
    ----------
        rswitch : Number or unit.Quantity
            The distance marking the start of the switching range.
        rcut : Number or unit.Quantity
            The potential cut-off distance.
        shifted : Bool, optional, default=True
            If True, a potential shift is done for the Coulomb term at the cutoff distance prior
            to smoothing.

    """
    def __init__(self, rswitch, rcut, shifted=True):
        potential = "4*epsilon*((sigma/r)^12-(sigma/r)^6) + Kc*charge1*charge2/r"
        shifting = "-(4*epsilon*((sigma/rc)^12-(sigma/rc)^6) + Kc*charge1*charge2/rc);"
        energy = potential + (shifting if shifted else ";")
        energy += "sigma = 0.5*(sigma1+sigma2);"
        energy += "epsilon = sqrt(epsilon1*epsilon2);"
        globalParameters = dict(Kc=138.935456*unit.kilojoules/unit.nanometer)
        if shifted:
            globalParameters["rc"] = rcut
        force = _CustomNonbondedForce(energy, rcut, rswitch, **globalParameters)
        super(InnerRespaForce, self).__init__([force])
        self.rswitch = rswitch
        self.rcut = rcut
        self.shifted = shifted


class OuterRespaForce(Force):
    """
    The outermost force for RESPA.

    Parameters
    ----------
        rswitch : Number or unit.Quantity
            The distance marking the start of the switching range.
        rcut : Number or unit.Quantity
            The potential cut-off distance.
        preceding : :class:`InnerRespaForce`
            The InnerRespaForce object with which this Force is supposed to match.

    """
    def __init__(self, rswitch, rcut, preceding):
        if not isinstance(preceding, InnerRespaForce):
            raise InputError("argument 'preceding' must be an internal RESPA force")
        potential = "-(4*epsilon*((sigma/r)^12-(sigma/r)^6) + Kc*charge1*charge2/r)"
        shifting = "+(4*epsilon*((sigma/rc)^12-(sigma/rc)^6) + Kc*charge1*charge2/rc);"
        energy = potential + (shifting if preceding.shifted else ";")
        energy += "sigma = 0.5*(sigma1+sigma2);"
        energy += "epsilon = sqrt(epsilon1*epsilon2);"
        globalParams = dict(Kc=138.935456*unit.kilojoules/unit.nanometer)
        if preceding.shifted:
            globalParams["rc"] = rcut
        discount = _CustomNonbondedForce(energy, preceding.rcut, preceding.rswitch, **globalParams)
        total = _NonbondedForce(rcut, rswitch)
        super(OuterRespaForce, self).__init__([total, discount])
