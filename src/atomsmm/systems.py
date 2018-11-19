"""
.. module:: system
   :platform: Unix, Windows
   :synopsis: a module for defining extensions of OpenMM System_ class.

.. moduleauthor:: Charlles R. A. Abreu <abreu@eq.ufrj.br>

.. _System: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.System.html

"""

import copy

from simtk import openmm

import atomsmm


class RESPASystem(openmm.System):
    """
    This is the base class of every AtomsMM System object.

    """
    def __init__(self, system, rcutIn, rswitchIn, **kwargs):
        self.__dict__ = copy.deepcopy(system).__dict__
        for force in self.getForces():
            force.setForceGroup(0)

        nonbonded = self.getForce(atomsmm.findNonbondedForce(self))

        exceptions = atomsmm.NonbondedExceptionsForce()
        exceptions.extractFrom(nonbonded)
        if exceptions.getNumBonds() > 0:
            self.addForce(exceptions)

        adjustment = kwargs.pop('adjustment', 'force-switch')
        innerForce = atomsmm.NearNonbondedForce(rcutIn, rswitchIn, adjustment)
        innerForce.importFrom(nonbonded)
        self.addForce(innerForce)

        potential = innerForce.expression.split(';')
        potential[0] = '-step(rc0-r)*({})'.format(potential[0])
        potential = ';'.join(potential)
        cutoff = nonbonded.getCutoffDistance()
        discount = atomsmm.forces._AtomsMM_CustomNonbondedForce(potential, cutoff, None, **innerForce.globalParams)
        discount.importFrom(nonbonded)
        self.addForce(discount)

        exceptions.setForceGroup(0)
        innerForce.setForceGroup(1)
        discount.setForceGroup(2)
        nonbonded.setForceGroup(2)
        nonbonded.setReciprocalSpaceForceGroup(2)
