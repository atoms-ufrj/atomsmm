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


class _AtomsMMSystem(openmm.System):
    def __init__(self, system, copyForces=True):
        super().__init__()
        self.setDefaultPeriodicBoxVectors(*system.getDefaultPeriodicBoxVectors())
        for index in range(system.getNumParticles()):
            self.addParticle(system.getParticleMass(index))
        for index in range(system.getNumParticles()):
            if system.isVirtualSite(index):
                self.setVirtualSite(index, system.getVirtualSite())
        for index in range(system.getNumConstraints()):
            self.addConstraint(*system.getConstraintParameters(index))
        if copyForces:
            for force in system.getForces():
                self.addForce(copy.deepcopy(force))


class RESPASystem(_AtomsMMSystem):
    """
    An OpenMM System_ prepared for Multiple Time-Scale Integration with RESPA.

    Parameters
    ----------
        system : openmm.System
            The original system from which to generate the RESPASystem.
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
        super().__init__(system)

        for force in self.getForces():
            if isinstance(force, openmm.NonbondedForce):
                nonbonded = force
                force.setForceGroup(2)
                force.setReciprocalSpaceForceGroup(2)
            else:
                force.setForceGroup(0)

        exceptions = atomsmm.NonbondedExceptionsForce()
        exceptions.extractFrom(nonbonded)
        if exceptions.getNumBonds() > 0:
            exceptions.setForceGroup(0)
            self.addForce(exceptions)

        adjustment = kwargs.pop('adjustment', 'force-switch')
        innerForce = atomsmm.NearNonbondedForce(rcutIn, rswitchIn, adjustment)
        innerForce.importFrom(nonbonded)
        innerForce.setForceGroup(1)
        self.addForce(innerForce)

        potential = innerForce.getEnergyFunction().split(';')
        potential[0] = '-step(rc0-r)*({})'.format(potential[0])
        potential = ';'.join(potential)
        cutoff = nonbonded.getCutoffDistance()
        globals = innerForce.getGlobalParameters()
        discount = atomsmm.forces._AtomsMM_CustomNonbondedForce(potential, cutoff, None, **globals)
        discount.importFrom(nonbonded)
        discount.setForceGroup(2)
        self.addForce(discount)


class VirialComputationSystem(_AtomsMMSystem):
    """
    An OpenMM System_ prepared for virial and pressure computation and reporting.

    Parameters
    ----------
        system : openmm.System
            The original system from which to generate the VirialComputationSystem.

    """
    def __init__(self, system, **kwargs):
        if system.getNumConstraints() > 0:
            raise RuntimeError('cannot compute virial/pressure for system with constraints')
        super().__init__(system, copyForces=False)
        for force in system.getForces():
            if isinstance(force, openmm.NonbondedForce) and force.getNumParticles() > 0:
                self.addForce(copy.deepcopy(force))
                expression = '4*epsilon*(11*(sigma/r)^12-5*(sigma/r)^6)'
                expression += '; sigma=0.5*(sigma1+sigma2)'
                expression += '; epsilon=sqrt(epsilon1*epsilon2)'
                rcut = force.getCutoffDistance()
                rswitch = force.getSwitchingDistance() if force.getUseSwitchingFunction() else None
                delta = atomsmm.forces._AtomsMM_CustomNonbondedForce(expression, rcut, rswitch, usesCharges=False)
                delta.importFrom(force)
                self.addForce(delta)
                exceptions = atomsmm.forces._AtomsMM_CustomBondForce(expression, usesCharges=False)
                exceptions.importFrom(force)
                if exceptions.getNumBonds() > 0:
                    self.addForce(exceptions)
            elif isinstance(force, openmm.HarmonicBondForce) and force.getNumBonds() > 0:
                bondforce = openmm.CustomBondForce('-K*r*(r-r0)')
                bondforce.addPerBondParameter('r0')
                bondforce.addPerBondParameter('K')
                for index in range(force.getNumBonds()):
                    i, j, r0, K = force.getBondParameters(index)
                    bondforce.addBond(i, j, [r0, K])
                self.addForce(bondforce)
            elif isinstance(force, openmm.CustomBondForce) and force.getNumBonds() > 0:
                raise RuntimeError("CustomBondForce support not implemented yet")
