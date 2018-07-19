"""
.. module:: utils
   :platform: Unix, Windows
   :synopsis: a module for auxiliary tasks.

.. moduleauthor:: Charlles R. A. Abreu <abreu@eq.ufrj.br>

"""

from simtk import openmm


class InputError(Exception):
    def __init__(self, msg):
        super(InputError, self).__init__("\033[1;31m" + msg + "\033[0m")


def FindNonbondedForce(system, position=0):
    """
    Searches for a NonbondedForce object attached to an OpenMM system.

    Parameters
    ----------
        system : openmm.System
            The system object in which the NonbondedForce will be searched for.
        position : int, optional, default=0
            The position index of the searched force among the NonbondedForce objects attached to
            the specified system.

    Returns
    -------
        force : openmm.NonbondedForce
            The retrieved NonbondedForce object.
        index : int
            The index of the NonbondedForce in the list of all forces attached to the system.

    """
    forces = [system.getForce(i) for i in range(system.getNumForces())]
    index = [i for (i, f) in enumerate(forces) if isinstance(f, openmm.NonbondedForce)][position]
    return system.getForce(index), index
