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
        system : System_
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
        inline : bool, optional, default=False
            If `True`, then the system passed as argument will be modified to become a RESPASystem.
            Otherwise, a new system will be created, leaving the passed one intact.

    """
    def __init__(self, system, rcutIn, rswitchIn, **kwargs):
        if kwargs.pop('inline', False):
            self.__dict__ = system.__dict__
        else:
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


class VirialComputationSystem(openmm.System):
    """
    An OpenMM System_ prepared for virial and pressure computation and reporting.

    Parameters
    ----------
        system : System_
            The original system from which to generate the VirialComputationSystem.

    Keyword Args
    ------------
        inline : bool, optional, default=False
            If `True`, then the system passed as argument will be modified to become a
            VirialComputationSystem. Otherwise, a new system will be created, leaving the passed
            one intact.

    """
    def __init__(self, system, **kwargs):
        if system.getNumConstraints() > 0:
            raise RuntimeError('cannot create VirialComputationSystem from a system with constraints')

        if kwargs.pop('inline', False):
            self.__dict__ = system.__dict__
        else:
            self.__dict__ = copy.deepcopy(system).__dict__

        def first(members, type):
            for (index, member) in enumerate(members):
                if isinstance(member, type):
                    return index, member
            return None, None

        iforce, nbforce = first(self.getForces(), openmm.NonbondedForce)
        if nbforce and nbforce.getNumParticles() > 0:
            expression = 'select(virialSwitch, 4*epsilon*(11*(sigma/r)^12-5*(sigma/r)^6), 0)'
            expression += '; sigma=0.5*(sigma1+sigma2)'
            expression += '; epsilon=sqrt(epsilon1*epsilon2)'
            force = openmm.CustomNonbondedForce(expression)
            force.addGlobalParameter('virialSwitch', 0)
            force.addPerParticleParameter('sigma')
            force.addPerParticleParameter('epsilon')
            mapping = {nbforce.CutoffNonPeriodic: force.CutoffNonPeriodic,
                       nbforce.CutoffPeriodic: force.CutoffPeriodic,
                       nbforce.Ewald: force.CutoffPeriodic,
                       nbforce.NoCutoff: force.NoCutoff,
                       nbforce.PME: force.CutoffPeriodic}
            force.setNonbondedMethod(mapping[nbforce.getNonbondedMethod()])
            force.setCutoffDistance(nbforce.getCutoffDistance())
            force.setUseLongRangeCorrection(nbforce.getUseDispersionCorrection())
            useSwitchingFunction = nbforce.getUseSwitchingFunction()
            force.setUseSwitchingFunction(useSwitchingFunction)
            if useSwitchingFunction:
                force.setSwitchingDistance(nbforce.getSwitchingDistance())
            for index in range(nbforce.getNumParticles()):
                charge, sigma, epsilon = nbforce.getParticleParameters(index)
                force.addParticle([sigma, epsilon])
            exceptions = openmm.CustomBondForce(expression)
            exceptions.addGlobalParameter('virialSwitch', 0)
            exceptions.addPerBondParameter('sigma')
            exceptions.addPerBondParameter('epsilon')
            for index in range(nbforce.getNumExceptions()):
                i, j, chargeprod, sigma, epsilon = nbforce.getExceptionParameters(index)
                force.addExclusion(i, j)
                if epsilon/epsilon.unit != 0.0:
                    exceptions.addBond(i, j, [sigma, epsilon])
            self.addForce(force)
            if exceptions.getNumBonds() > 0:
                self.addForce(exceptions)

        iforce, bondforce = first(self.getForces(), openmm.HarmonicBondForce)
        if bondforce and bondforce.getNumBonds() > 0:
            expression = 'select(virialSwitch, -K*r*(r-r0), 0.5*K*(r-r0)^2)'
            force = openmm.CustomBondForce(expression)
            force.addGlobalParameter('virialSwitch', 0)
            force.addPerBondParameter('r0')
            force.addPerBondParameter('K')
            for index in range(bondforce.getNumBonds()):
                i, j, r0, K = bondforce.getBondParameters(index)
                force.addBond(i, j, [r0, K])
            self.removeForce(iforce)
            self.addForce(force)

        iforce, angleforce = first(self.getForces(), openmm.HarmonicAngleForce)
        if angleforce and angleforce.getNumAngles() > 0:
            expression = 'select(virialSwitch, 0, 0.5*K*(theta-theta0)^2)'
            force = openmm.CustomAngleForce(expression)
            force.addGlobalParameter('virialSwitch', 0)
            force.addPerAngleParameter('theta0')
            force.addPerAngleParameter('K')
            for index in range(angleforce.getNumAngles()):
                i, j, k, theta0, K = angleforce.getAngleParameters(index)
                force.addAngle(i, j, k, [theta0, K])
            self.removeForce(iforce)
            self.addForce(force)

        iforce, torsionforce = first(self.getForces(), openmm.PeriodicTorsionForce)
        if torsionforce and torsionforce.getNumTorsions() > 0:
            expression = 'select(virialSwitch, 0, K*(1+cos(n*theta−theta0)))'
            force = openmm.CustomTorsionForce(expression)
            force.addGlobalParameter('virialSwitch', 0)
            force.addPerTorsionParameter('n')
            force.addPerTorsionParameter('theta0')
            force.addPerTorsionParameter('K')
            for index in range(torsionforce.getNumAngles()):
                i, j, k, l, n, theta0, K = torsionforce.getTorsionParameters(index)
                force.addTorsion(i, j, k, l, [n, theta0, K])
            self.removeForce(iforce)
            self.addForce(force)
