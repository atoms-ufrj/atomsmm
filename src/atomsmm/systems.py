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
        self.this = new_system.this


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
        solute_atoms : set(int)
            A set containing the indexes of all solute atoms.
        respa_info : dict(str:unit.Quantity), optional, default=None
            Parameters for splitting the forces into force groups aiming at multiple time-scale
            integration with a RESPA scheme. If this is `None`, then no splitting will be done.
            Otherwise, a dictionary with mandatory keywords 'rcutIn' and 'rswitchIn' and optional
            keyword 'adjustment' must be passed. These are explained in :class:`RESPASystem`.

    """
    def __init__(self, system, solute_atoms, respa_info=None):
        solution = copy.deepcopy(system)

        # Separate the nonbonded force from other pre-existing forces:
        other_forces = []
        for force in solution.getForces():
            if isinstance(force, openmm.NonbondedForce):
                nonbonded = force
            else:
                other_forces.append(force)

        # Store general system properties and define auxiliary function:
        rcut = nonbonded.getCutoffDistance()
        rswitch = nonbonded.getSwitchingDistance() if nonbonded.getUseSwitchingFunction() else None
        all_atoms = set(range(nonbonded.getNumParticles()))
        solvent_atoms = all_atoms - solute_atoms

        # A custom nonbonded force for solute-solvent, softcore van der Waals interactions:
        softcore = atomsmm.SoftcoreLennardJonesForce(rcut, rswitch, parameter='lambda_vdw')
        softcore.importFrom(nonbonded)
        softcore.addInteractionGroup(solute_atoms, solvent_atoms)
        solution.addForce(softcore)

        # All solute-solute interactions are treated as nonbonded exceptions:
        exception_pairs = []
        for index in range(nonbonded.getNumExceptions()):
            i, j, _, _, _ = nonbonded.getExceptionParameters(index)
            if set([i, j]).issubset(solute_atoms):
                exception_pairs.append(set([i, j]))
        for i, j in itertools.combinations(solute_atoms, 2):
            if set([i, j]) not in exception_pairs:
                q1, sig1, eps1 = nonbonded.getParticleParameters(i)
                q2, sig2, eps2 = nonbonded.getParticleParameters(j)
                nonbonded.addException(i, j, q1*q2, (sig1 + sig2)/2, np.sqrt(eps1*eps2))

        # Turn off intrasolute Lennard-Jones interactions and scale solute charges by lambda_coul:
        charges = dict()
        for index in solute_atoms:
            charge, _, _ = nonbonded.getParticleParameters(index)
            nonbonded.setParticleParameters(index, 0.0, 1.0, 0.0)
            if charge/charge.unit != 0.0:
                charges[index] = charge
        if charges:
            nonbonded.addGlobalParameter('lambda_coul', 1.0)
            for index, charge in charges.items():
                nonbonded.addParticleParameterOffset('lambda_coul', index, charge, 0.0, 0.0)

        # Add short-ranged nonbonded force if respa info has been passed:
        if respa_info is not None:
            for force in other_forces:
                force.setForceGroup(0)
            softcore.setForceGroup(1)
            nonbonded.setForceGroup(2)
            nonbonded.setReciprocalSpaceForceGroup(2)

            rcutIn = respa_info['rcutIn']
            rswitchIn = respa_info['rswitchIn']
            adjustment = respa_info.get('adjustment', 'force-switch')

            respa_system = RESPASystem(solution, rcutIn, rswitchIn, adjustment=adjustment, keepGroups=True)

            if charges:
                def add_force(expressions, group):
                    force = openmm.CustomNonbondedForce(';'.join(expressions))
                    force.setForceGroup(group)
                    force.setCutoffDistance(rcutIn if group == 1 else rcut)
                    force.addInteractionGroup(solute_atoms, solvent_atoms)
                    force.addGlobalParameter('lambda_coul', 1.0)
                    force.addPerParticleParameter('charge')
                    force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
                    for i in range(nonbonded.getNumParticles()):
                        charge, _, _ = nonbonded.getParticleParameters(i)
                        force.addParticle([charge])
                    for index in range(nonbonded.getNumParticleParameterOffsets()):
                        _, i, charge, _, _ = nonbonded.getParticleParameterOffset(index)
                        force.setParticleParameters(i, [charge])
                    for index in range(nonbonded.getNumExceptions()):
                        i, j, _, _, _ = nonbonded.getExceptionParameters(index)
                        force.addExclusion(i, j)
                    respa_system.addForce(force)

                nearForce = atomsmm.forces.nearForceExpressions(rcutIn, rswitchIn, adjustment)
                minusNearForce = copy.deepcopy(nearForce)
                minusNearForce[0] = '-step(rc0-r)*({})'.format(nearForce[0])
                mixing_rule = ['sigma=1', 'epsilon=0', 'chargeprod=lambda_coul*charge1*charge2']
                add_force(nearForce + mixing_rule, 1)
                add_force(minusNearForce + mixing_rule, 2)

            super().__init__(respa_system)
        else:
            super().__init__(solution)


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
