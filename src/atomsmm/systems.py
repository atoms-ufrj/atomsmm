"""
.. module:: system
   :platform: Unix, Windows
   :synopsis: a module for defining extensions of OpenMM System_ class.

.. moduleauthor:: Charlles R. A. Abreu <abreu@eq.ufrj.br>

.. _System: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.System.html

"""

import copy
import itertools

import numpy as np
from simtk import openmm
from sympy import Symbol
from sympy.parsing.sympy_parser import parse_expr

import atomsmm


class _AtomsMM_System(openmm.System):
    def __init__(self, system, copyForces=True):
        new_system = copy.deepcopy(system)
        if not copyForces:
            for index in reversed(range(new_system.getNumForces())):
                new_system.removeForce(index)
        self.__dict__ = new_system.__dict__


class RESPASystem(_AtomsMM_System):
    """
    An OpenMM System_ prepared for Multiple Time-Scale Integration with RESPA.

    Parameters
    ----------
        system : openmm.System
            The original system from which to generate the RESPASystem.
        rcutIn : unit.Quantity
            The distance at which the short-range nonbonded interactions will completely vanish.
        rswitchIn : unit.Quantity
            The distance at which the short-range nonbonded interactions will start vanishing by
            application of a switching function.

    Keyword Args
    ------------
        adjustment : str, optional, default='force-switch'
            A keyword for modifying the near nonbonded potential energy function. If it is `None`,
            then the switching function is applied directly to the original potential. Other options
            are `'shift'` and `'force-switch'`. If it is `'shift'`, then the switching function is
            applied to a potential that is already null at the cutoff due to a previous shift.
            If it is `'force-switch'`, then the potential is modified so that the switching
            function is applied to the forces rather than the potential energy.
        fastExceptions : bool, optional, default=True
            Whether nonbonded exceptions must be considered to belong to the group of fastest
            forces. If `False`, then they will be split into intermediate and slowest forces.
        keepGroups : bool, optional, default=False
            Whether forge groups should be kept unchanged (except for nonbonded forces).

    """
    def __init__(self, system, rcutIn, rswitchIn, **kwargs):
        adjustment = kwargs.pop('adjustment', 'force-switch')
        fastExceptions = kwargs.get('fastExceptions', True)
        keepGroups = kwargs.get('keepGroups', False)
        super().__init__(system)
        for force in self.getForces():
            if isinstance(force, openmm.NonbondedForce):
                # Every nonbonded force is placed in the longest time scale (group 2):
                force.setForceGroup(2)
                force.setReciprocalSpaceForceGroup(2)

                # For each nonbonded force, a short-ranged version is created and allocated in the
                # intermediary time scale (group 1):
                innerForce = atomsmm.NearNonbondedForce(rcutIn, rswitchIn, adjustment)
                innerForce.importFrom(force)  # Non-exclusion exceptions become exclusions here
                innerForce.setForceGroup(1)
                self.addForce(innerForce)

                # The same short-ranged version is subtracted from the slowest forces (group 2):
                negative = atomsmm.NearNonbondedForce(rcutIn, rswitchIn, adjustment, subtract=True,
                                                      actual_cutoff=force.getCutoffDistance())
                negative.importFrom(force)  # Non-exclusion exceptions become exclusions here
                negative.setForceGroup(2)
                self.addForce(negative)

                if fastExceptions:
                    # Non-exclusion exceptions (if any) are extracted from the nonbonded force and
                    # placed in the shortest time scale:
                    exceptions = atomsmm.NonbondedExceptionsForce()
                    exceptions.importFrom(force, extract=True)
                    if exceptions.getNumBonds() > 0:
                        exceptions.setForceGroup(0)
                        self.addForce(exceptions)
                else:
                    # A short-ranged version of each non-exclusion exception (if any) is added to
                    # the intermediary time scale and subtracted from the slowest one:
                    exceptions = atomsmm.NearExceptionForce(rcutIn, rswitchIn, adjustment)
                    exceptions.importFrom(force)
                    if exceptions.getNumBonds() > 0:
                        exceptions.setForceGroup(1)
                        self.addForce(exceptions)
                        negative = atomsmm.NearExceptionForce(rcutIn, rswitchIn, adjustment, subtract=True)
                        negative.importFrom(force)
                        negative.setForceGroup(2)
                        self.addForce(negative)
            elif not keepGroups:
                # All other forces are allocated in the shortest time scale (group 0):
                force.setForceGroup(0)


class SolvationSystem(_AtomsMM_System):
    """
    An OpenMM System_ prepared for solvation free-energy calculations.

    Rules:

    Parameters
    ----------
        system : openmm.System
            The original system from which to generate the SolvationSystem.
        soluteAtoms : set(int)
            A set containing the indexes of all solute atoms.
        respaInfo : dict(str:unit.Quantity), optional, default=None
            Parameters for splitting the forces into force groups aiming at multiple time-scale
            integration with a RESPA scheme. If this is `None`, then no splitting will be done.
            Otherwise, a dictionary with mandatory keywords 'rcutIn' and 'rswitchIn' and optional
            keyword 'adjustment' must be passed. These are explained in :class:`RESPASystem`.
        forceGroup : int, optional, default=0
            The force group in which to include the new solute-solute and solute-solvent forces.

    """
    def __init__(self, system, soluteAtoms, respaInfo=None, forceGroup=0):
        super().__init__(system)

        # Separate the nonbonded force from other pre-existing forces:
        other_forces = []
        for force in self.getForces():
            if isinstance(force, openmm.NonbondedForce):
                nonbonded = force
            else:
                other_forces.append(force)

        # Store general system properties:
        rcut = nonbonded.getCutoffDistance()
        rswitch = nonbonded.getSwitchingDistance() if nonbonded.getUseSwitchingFunction() else None
        allAtoms = set(range(nonbonded.getNumParticles()))
        solventAtoms = allAtoms - soluteAtoms

        # A custom nonbonded force for solute-solvent, softcore van der Waals interactions:
        softcore = atomsmm.SoftcoreLennardJonesForce(rcut, rswitch, parameter='lambda_vdw')
        softcore.importFrom(nonbonded)
        softcore.addInteractionGroup(soluteAtoms, solventAtoms)

        # A custom bond force for holding full exceptions:
        exceptions = openmm.CustomBondForce('4*epsilon*x*(x-1) + Kc*chargeprod/r; x=(sigma/r)^6; Kc=138.935456')
        for parameter in ['chargeprod', 'sigma', 'epsilon']:
            exceptions.addPerBondParameter(parameter)

        # Custom bond forces for holding solute-solute recoupling interactions:
        recoupling_atom_atom = openmm.CustomBondForce('(1-lambda_coul)^2*Kc*chargeprod/r; Kc=138.935456')
        recoupling_exception = openmm.CustomBondForce('(1-lambda_coul)*Kc*chargeprod/r; Kc=138.935456')
        for force in [recoupling_atom_atom, recoupling_exception]:
            force.addPerBondParameter('chargeprod')
            force.addGlobalParameter('lambda_coul', 1.0)

        def nonzero(quantity):
            return quantity/quantity.unit != 0.0

        # Remove exceptions from the nonbonded force, distribute them to the new forces:
        exception_pair = dict()
        exception_charge_prod = dict()
        for index in range(nonbonded.getNumExceptions()):
            i, j, chargeprod, sigma, epsilon = nonbonded.getExceptionParameters(index)
            nonbonded.setExceptionParameters(index, i, j, 0.0, 1.0, 0.0)
            pair = set([i, j])
            if pair.issubset(solventAtoms):
                if nonzero(chargeprod) or nonzero(epsilon):
                    exceptions.addBond(i, j, [chargeprod, sigma, epsilon])
            elif pair.issubset(soluteAtoms):
                exception_pair[index] = pair
                exception_charge_prod[index] = chargeprod
                if nonzero(chargeprod):
                    recoupling_exception.addBond(i, j, [chargeprod])
                if nonzero(epsilon):
                    exceptions.addBond(i, j, [0.0, sigma, epsilon])
            else:
                raise Exception('Solute-solvent exceptions cannot be handled')

        # Add all other nonbonded solute-solute interactions to the new forces:
        for i, j in itertools.combinations(soluteAtoms, 2):
            if set([i, j]) not in exception_pair.values():
                q1, sig1, eps1 = nonbonded.getParticleParameters(i)
                q2, sig2, eps2 = nonbonded.getParticleParameters(j)
                if nonzero(q1) and nonzero(q2):
                    recoupling_atom_atom.addBond(i, j, [q1*q2])
                if nonzero(eps1) and nonzero(eps2):
                    exceptions.addBond(i, j, [0.0, (sig1 + sig2)/2, np.sqrt(eps1*eps2)])

        # Add the new forces to the system:
        for force in [softcore, exceptions, recoupling_atom_atom, recoupling_exception]:
            self.addForce(force)

        # Add short-ranged nonbonded force if respa info has been passed:
        if respaInfo is not None:
            rcutIn = respaInfo['rcutIn']
            rswitchIn = respaInfo['rswitchIn']
            adjustment = respaInfo.get('adjustment', 'force-switch')

            nearForce = atomsmm.forces.nearForceExpressions(rcutIn, rswitchIn, adjustment)
            minusNearForce = copy.deepcopy(nearForce)
            minusNearForce[0] = '-step(rc0-r)*({})'.format(nearForce[0])

            lj_mixing = ['sigma=(sigma1+sigma2)/2', 'epsilon=sqrt(epsilon1*epsilon2)']
            solvent_mixing = lj_mixing + ['chargeprod=charge1*charge2']
            solute_mixing = lj_mixing + ['chargeprod=lambda_coul^2*charge1*charge2']

            solvent_inner = openmm.CustomNonbondedForce(';'.join(nearForce + solvent_mixing))
            solvent_negative = openmm.CustomNonbondedForce(';'.join(minusNearForce + solvent_mixing))
            solute_inner = openmm.CustomNonbondedForce(';'.join(nearForce + solute_mixing))
            solute_negative = openmm.CustomNonbondedForce(';'.join(minusNearForce + solute_mixing))

            solvent_inner.setCutoffDistance(rcutIn)
            solute_inner.setCutoffDistance(rcutIn)
            solvent_negative.setCutoffDistance(rcut)
            solute_negative.setCutoffDistance(rcut)

            solvent_inner.addInteractionGroup(solventAtoms, solventAtoms)
            solvent_negative.addInteractionGroup(solventAtoms, solventAtoms)
            solute_inner.addInteractionGroup(soluteAtoms, allAtoms)
            solute_negative.addInteractionGroup(soluteAtoms, allAtoms)

            solute_inner.addGlobalParameter('lambda_coul', 1.0)
            solute_negative.addGlobalParameter('lambda_coul', 1.0)

            for force in [solvent_inner, solvent_negative, solute_inner, solute_negative]:
                for parameter in ['charge', 'sigma', 'epsilon']:
                    force.addPerParticleParameter(parameter)
                force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
                for i in range(nonbonded.getNumParticles()):
                    force.addParticle(nonbonded.getParticleParameters(i))
                for index in range(nonbonded.getNumExceptions()):
                    i, j, _, _, _ = nonbonded.getExceptionParameters(index)
                    force.addExclusion(i, j)
                self.addForce(force)

            nonbonded.setForceGroup(2)
            nonbonded.setReciprocalSpaceForceGroup(2)
            for force in other_forces:
                force.setForceGroup(0)
            softcore.setForceGroup(1)
            exceptions.setForceGroup(0)
            solvent_inner.setForceGroup(1)
            solvent_negative.setForceGroup(2)
            recoupling_atom_atom.setForceGroup(2)
            recoupling_exception.setForceGroup(2)

        # Delete van der Waals parameters and scale charges of solute atoms:
        nonbonded.addGlobalParameter('lambda_coul', 1.0)
        for index in soluteAtoms:
            charge, _, _ = nonbonded.getParticleParameters(index)
            nonbonded.setParticleParameters(index, 0.0, 1.0, 0.0)
            if nonzero(charge):
                nonbonded.addParticleParameterOffset('lambda_coul', index, charge, 0.0, 0.0)
        for index, chargeprod in exception_charge_prod.items():
            if nonzero(chargeprod):
                nonbonded.addExceptionParameterOffset('lambda_coul', index, chargeprod, 0.0, 0.0)


class ComputingSystem(_AtomsMM_System):
    """
    An OpenMM System_ prepared for computing the Coulomb contribution to the potential energy, as
    well as the total internal virial of an atomic system.

    ..warning:
        Currently, virial computation is only supported for fully flexible systems (i.e. without
        distance constraints).

    Parameters
    ----------
        system : openmm.System
            The original system from which to generate the ComputingSystem.

    """
    def __init__(self, system, **kwargs):
        super().__init__(system, copyForces=False)
        if system.getNumConstraints() > 0:
            raise RuntimeError('virial/pressure computation not supported for system with constraints')
        dispersionGroup = 0
        bondedGroup = 1
        coulombGroup = 2
        othersGroup = 3
        self._dispersion = 2**dispersionGroup
        self._bonded = 2**bondedGroup
        self._coulomb = 2**coulombGroup
        self._others = 2**othersGroup
        for force in system.getForces():
            if isinstance(force, openmm.NonbondedForce) and force.getNumParticles() > 0:
                nonbonded = copy.deepcopy(force)
                if nonbonded.getUseDispersionCorrection():
                    raise RuntimeError('virial/pressure computation not supported for force with dispersion correction')
                nonbonded.setForceGroup(coulombGroup)
                nonbonded.setReciprocalSpaceForceGroup(coulombGroup)
                self.addForce(nonbonded)
                expression = '24*epsilon*(2*(sigma/r)^12-(sigma/r)^6)'
                expression += '; sigma=0.5*(sigma1+sigma2)'
                expression += '; epsilon=sqrt(epsilon1*epsilon2)'
                rcut = force.getCutoffDistance()
                rswitch = force.getSwitchingDistance() if force.getUseSwitchingFunction() else None
                virial = atomsmm.forces._AtomsMM_CustomNonbondedForce(expression, rcut, rswitch, charged=False)
                virial.importFrom(nonbonded)
                virial.setForceGroup(dispersionGroup)
                self.addForce(virial)
                exceptions = atomsmm.forces._AtomsMM_CustomBondForce(expression, charged=False)
                for index in range(nonbonded.getNumExceptions()):
                    i, j, chargeprod, sigma, epsilon = nonbonded.getExceptionParameters(index)
                    if epsilon/epsilon.unit != 0.0:
                        exceptions.addBond(i, j, [sigma, epsilon])
                        nonbonded.setExceptionParameters(index, i, j, chargeprod, 1.0, 0.0)
                if exceptions.getNumBonds() > 0:
                    exceptions.setForceGroup(dispersionGroup)
                    self.addForce(exceptions)
                for index in range(nonbonded.getNumParticles()):
                    charge, sigma, epsilon = nonbonded.getParticleParameters(index)
                    nonbonded.setParticleParameters(index, charge, 1.0, 0.0)
            elif isinstance(force, openmm.HarmonicBondForce) and force.getNumBonds() > 0:
                bondforce = openmm.CustomBondForce('-K*r*(r-r0)')
                bondforce.addPerBondParameter('r0')
                bondforce.addPerBondParameter('K')
                for index in range(force.getNumBonds()):
                    i, j, r0, K = force.getBondParameters(index)
                    bondforce.addBond(i, j, [r0, K])
                bondforce.setForceGroup(bondedGroup)
                self.addForce(bondforce)
            elif isinstance(force, openmm.CustomBondForce) and force.getNumBonds() > 0:
                bondforce = openmm.CustomBondForce(self._virialExpression(force))
                for index in range(force.getNumPerBondParameters()):
                    bondforce.addPerBondParameter(force.getPerBondParameterName(index))
                for index in range(force.getNumGlobalParameters()):
                    bondforce.addGlobalParameter(force.getGlobalParameterName(index),
                                                 force.getGlobalParameterDefaultValue(index))
                for index in range(force.getNumBonds()):
                    bondforce.addBond(*force.getBondParameters(index))
                bondforce.setForceGroup(bondedGroup)
                self.addForce(bondforce)
            else:
                otherforce = copy.deepcopy(force)
                otherforce.setForceGroup(othersGroup)
                self.addForce(otherforce)

    def _virialExpression(self, force):
        definitions = force.getEnergyFunction().split(';')
        function = parse_expr(definitions.pop(0))
        for definition in definitions:
            name, expression = definition.split('=')
            symbol = Symbol(name.strip())
            expression = parse_expr(expression.replace('^', '**'))
            function = function.subs(symbol, expression)
        r = Symbol('r')
        virial = -r*function.diff(r)
        return virial.__repr__()
