"""
.. module:: forces
   :platform: Unix, Windows
   :synopsis: a module for defining the class :class:`force`.

.. moduleauthor:: Charlles R. A. Abreu <abreu@eq.ufrj.br>

"""

from simtk import openmm
from simtk import unit


class CustomNonbondedForce(openmm.CustomNonbondedForce):
    """
    An extension of OpenMM's CustomNonbondedForce_ class.

    .. _CustomNonbondedForce: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.CustomNonbondedForce.html

    """
    def __init__(self, energy):
        super(CustomNonbondedForce, self).__init__(energy)
        self.addPerParticleParameter("charge")
        self.addPerParticleParameter("sigma")
        self.addPerParticleParameter("epsilon")

    def addTo(self, system, capture=True, replace=False):
        """
        Add the nonbonded force to an OpenMM System_ object.

        .. _System: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.System.html

        Parameters
        ----------
            system : openmm.System
                The system to which the nonbonded force is being added.
            capture : Bool, optional, default=True
                If True, the added nonbonded force will capture all particles and exceptions of the
                system's first nonbonded force, if any.
            replace : Bool, optional, default=False
                If True, the added nonbonded force will replace the system's first nonbonded force,
                if any.

        """
        forces = [system.getForce(i) for i in range(system.getNumForces())]
        nbforces = [i for (i, f) in enumerate(forces) if isinstance(f, openmm.NonbondedForce)]
        if capture and nbforces:
            force = system.getForce(nbforces[0])
            for index in range(force.getNumParticles()):
                self.addParticle(force.getParticleParameters(index))
            for index in range(force.getNumExceptions()):
                i, j, chargeProd, sigma, epsilon = force.getExceptionParameters(index)
                self.addExclusion(i, j)
                if chargeProd/chargeProd.unit != 0.0 or epsilon/epsilon.unit != 0.0:
                    # TODO: Add bond force for handling 1-4 interactions
                    raise ValueError("Non-exclusion exceptions not handled yet.")
        if replace and nbforces:
            system.removeForce(nbforces[0])
        system.addForce(self)
        return self


class DampedSmoothedForce(CustomNonbondedForce):
    """
    A damped-smoothed version of the Lennard-Jones/Coulomb potential.

    .. math::
        & V(r)=\\theta(r_\\mathrm{cut}-r)\\left\\{
                   4\\epsilon\\left[\\left(\\frac{\\sigma}{r}\\right)^{12}-\\left(\\frac{\\sigma}{r}\\right)^6\\right]
                   + \\frac{q_1 q_2}{4\\pi\\epsilon_0}\\frac{\\mathrm{erfc}(\\alpha r)}{r}
               \\right\\}f(r) \\\\
        & f(r)=[1+\\theta(r-r_\\mathrm{switch})u^3(15u-6u^2-10)] \\\\
        & u=\\frac{r^n-r_\\mathrm{switch}^n}{r_\\mathrm{cut}^n-r_\\mathrm{switch}^n} \\\\
        & \\sigma=\\frac{\\sigma_1+\\sigma_2}{2} \\\\
        & \\epsilon=\\sqrt{\\epsilon_1\\epsilon_2}

    In the equations above, :math:`\\theta(x)` is the Heaviside step function. Note that the
    switching function employed here, with `u` being a quadratic function of `r`, is slightly
    different from the one normally used in OpenMM, in which `u` is a linear function of `r`.

    Parameters
    ----------
        alpha
            The damping parameter (in inverse distance unit).
        rswitch
            The distance marking the start of the switching range.
        rcut
            The potential cut-off distance.
        degree : int, optional, default=1
            The degree `n` in the definition of the switching variable `u` (see above).

    """

    def __init__(self, alpha, rswitch, rcut, degree=1):

        # Model expressions:
        energy = "(4*epsilon*((sigma/r)^12-(sigma/r)^6) + Kcoul*charge1*charge2*erfc(alpha*r)/r)*f;"
        if degree == 1:
            energy += "f = 1;"
        else:
            energy += "f = 1 + step(r - rswitch)*u^3*(15*u - 6*u^2 - 10);"
            energy += "u = (r^%d - rswitch^%d)/(rcut^%d - rswitch^%d);" % ((degree,)*4)
        energy += "sigma = 0.5*(sigma1+sigma2);"
        energy += "epsilon = sqrt(epsilon1*epsilon2);"
        super(DampedSmoothedForce, self).__init__(energy)

        # Global parameters:
        self.addGlobalParameter("Kcoul", 138.935456*unit.kilojoules/unit.nanometer)
        self.addGlobalParameter("alpha", alpha)
        self.addGlobalParameter("rswitch", rswitch)
        self.addGlobalParameter("rcut", rcut)

        # Configuration:
        self.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
        self.setCutoffDistance(rcut)
        self.setUseLongRangeCorrection(False)
        if degree == 1:
            self.setUseSwitchingFunction(True)
            self.setSwitchingDistance(rswitch)
        else:
            self.setUseSwitchingFunction(False)
