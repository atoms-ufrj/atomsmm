"""
.. module:: utils
   :platform: Unix, Windows
   :synopsis: a module for auxiliary tasks.

.. moduleauthor:: Charlles R. A. Abreu <abreu@eq.ufrj.br>

"""

from copy import deepcopy

from simtk import openmm


class InputError(Exception):
    def __init__(self, msg):
        super(InputError, self).__init__("\033[1;31m" + msg + "\033[0m")


def LennardJones(r):
    return "4*epsilon*((sigma/%s)^12 - (sigma/%s)^6)" % (r, r)


def Coulomb(r):
    return "Kc*chargeprod/%s" % r


def LennardJonesCoulomb(r):
    return "%s + %s" % (LennardJones(r), Coulomb(r))


def LorentzBerthelot():
    mixingRule = "chargeprod = charge1*charge2;"
    mixingRule += "sigma = 0.5*(sigma1+sigma2);"
    mixingRule += "epsilon = sqrt(epsilon1*epsilon2);"
    return mixingRule


def HijackNonbondedForce(system, position=0):
    """
    Searches for and extracts a NonbondedForce object from an OpenMM system.

    .. warning::

        Side-effect: the passed system object will no longer have the hijacked NonbondedForce in
        its force list.

    Parameters
    ----------
        system : openmm.System
            The system to which the wanted NonbondedForce object is attached.
        position : int, optional, default=0
            The position index of the wanted force among the NonbondedForce objects attached to
            the system.

    Returns
    -------
        openmm.NonbondedForce
            The hijacked NonbondedForce object.

    """
    forces = [system.getForce(i) for i in range(system.getNumForces())]
    index = [i for (i, f) in enumerate(forces) if isinstance(f, openmm.NonbondedForce)][position]
    force = deepcopy(system.getForce(index))
    system.removeForce(index)
    return force
