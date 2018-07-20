"""
.. module:: utils
   :platform: Unix, Windows
   :synopsis: a module for auxiliary tasks.

.. moduleauthor:: Charlles R. A. Abreu <abreu@eq.ufrj.br>

"""

from copy import deepcopy

from simtk import openmm


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
