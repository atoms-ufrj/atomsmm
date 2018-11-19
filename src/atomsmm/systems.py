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
    An OpenMM System_ prepared for Multiple Time-Scale Integration with RESPA.

    Parameters
    ----------
        rcutIn : Number or unit.Quantity
            The distance at which the near nonbonded interactions vanish.
        rswitchIn : Number or unit.Quantity
            The distance at which the switching function begins to smooth the approach of the
            near nonbonded interaction towards zero.

    Keyword Args
    ------------
        adjustment : str, optional, default=None
            A keyword for modifying the near nonbonded potential energy function. If it is `None`,
            then the switching function is applied directly to the original potential. Other options
            are `'shift'` and `'force-switch'`. If it is `'shift'`, then the switching function is
            applied to a potential that is already null at the cutoff due to a previous shift.
            If it is `'force-switch'`, then the potential is modified so that the switching
            function is applied to the forces rather than the potential energy.

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

        potential = innerForce.getEnergyFunction().split(';')
        potential[0] = '-step(rc0-r)*({})'.format(potential[0])
        potential = ';'.join(potential)
        cutoff = nonbonded.getCutoffDistance()
        globals = innerForce.getGlobalParameters()
        discount = atomsmm.forces._AtomsMM_CustomNonbondedForce(potential, cutoff, None, **globals)
        discount.importFrom(nonbonded)
        self.addForce(discount)

        exceptions.setForceGroup(0)
        innerForce.setForceGroup(1)
        discount.setForceGroup(2)
        nonbonded.setForceGroup(2)
        nonbonded.setReciprocalSpaceForceGroup(2)
