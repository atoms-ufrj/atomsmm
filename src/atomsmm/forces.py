"""
.. module:: forces
   :platform: Unix, Windows
   :synopsis: a module for defining the class :class:`force`.

.. moduleauthor:: Charlles R. A. Abreu <abreu@eq.ufrj.br>

"""

import simtk.openmm as openmm

from atomsmm.utils import InputError


class CustomNonbondedForce(openmm.CustomNonbondedForce):
    """
    An extension of OpenMM's CustomNonbondedForce_ class.

    .. _CustomNonbondedForce: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.CustomNonbondedForce.html

    """

    def importParameters(self, force):
        print("imported")


class DampedSmoothedForce(CustomNonbondedForce):
    """
    A damped-smoothed version of the Lennard-Jones/Coulomb potential.

    .. math::
        & V(r)=\\theta(r_\\mathrm{cut}-r)\\left\\{
                   4\\epsilon\\left[\\left(\\frac{\\sigma}{r}\\right)^{12}-\\left(\\frac{\\sigma}{r}\\right)^6\\right]
                   + \\frac{q_1 q_2}{4\\pi\\epsilon_0}\\frac{\\mathrm{erfc}(\\alpha r)}{r}
               \\right\\}f(r) \\\\
        & f(r)=[1-\\theta(r-r_\\mathrm{switch})z^3(10-15z+6z^2)] \\\\
        & z=\\frac{r^2-r_\\mathrm{switch}^2}{r_\\mathrm{cut}^2-r_\\mathrm{switch}^2} \\\\
        & \\sigma=\\frac{\\sigma_1+\\sigma_2}{2} \\\\
        & \\epsilon=\\sqrt{\\epsilon_1\\epsilon_2}

    In the equations above, :math:`\\theta(x)` is the Heaviside step function. Note that the
    switching function employed here, with `z` being a quadratic function of `r`, is slightly
    different from the one normally used in OpenMM, in which `z` is a linear function of `r`.

    Parameters
    ----------
        alpha : Number
            The damping parameter (in inverse distance unit).
        rswitch : Number
            The distance marking the start of the switching range.
        rcut : Number
            The potential cut-off distance.

    """

    def __init__(self, alpha, rswitch, rcut):

        # Model expressions:
        energy = "(4*epsilon*((sigma/r)^12-(sigma/r)^6) + Kcoul*charge1*charge2*erfc(alpha*r)/r)*f;"
        energy += "sigma = (sigma1+sigma2)/2";
        energy += "epsilon = sqrt(epsilon1*epsilon2);"
        energy += "f = 1 - step(r - rswitch)*z^3*(10 - 15*z + 6*z^2);"
        energy += "z = (r^2 - rswitch^2)/(rcut^2 - rswitch^2);"
        super(CustomNonbondedForce, self).__init__(energy)

        # Per-particle parameters:
        self.addPerParticleParameter("epsilon")
        self.addPerParticleParameter("sigma")
        self.addPerParticleParameter("charge")

        # Global parameters:
        self.addGlobalParameter("Kcoul", 138.935456)
        self.addGlobalParameter("alpha", alpha)
        self.addGlobalParameter("rswitch", rswitch)
        self.addGlobalParameter("rcut", rcut)

        # Configuration:
        self.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
        self.setCutoffDistance(rcut)
        self.setUseLongRangeCorrection(False)
        self.setUseSwitchingFunction(False)
