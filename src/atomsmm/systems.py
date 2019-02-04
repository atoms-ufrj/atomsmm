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
        self.this = copy.deepcopy(system).this
        if not copyForces:
            for index in reversed(range(self.getNumForces())):
                self.removeForce(index)


class RESPASystem(openmm.System):
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

    """
    def __init__(self, system, rcutIn, rswitchIn, **kwargs):
        self.this = copy.deepcopy(system).this
        adjustment = kwargs.pop('adjustment', 'force-switch')
        fastExceptions = kwargs.get('fastExceptions', True)

        ljc_potential = ['4*epsilon*x*(x-1) + Kc*chargeprod/r', 'x=(sigma/r)^6', 'Kc=138.935456']
        near_potential = atomsmm.forces.nearForceExpressions(rcutIn, rswitchIn, adjustment)
        minus_near_potential = copy.deepcopy(near_potential)
        minus_near_potential[0] = '-step(rc0-r)*({})'.format(near_potential[0])

        for force in self.getForces():
            if isinstance(force, openmm.NonbondedForce):
                rcut = force.getCutoffDistance()
                force.setForceGroup(2)
                force.setReciprocalSpaceForceGroup(2)
                self._addCustomNonbondedForce(near_potential, rcutIn, 1, force)
                self._addCustomNonbondedForce(minus_near_potential, rcut, 2, force)
                if fastExceptions:
                    self._addCustomBondForce(ljc_potential, 0, force, extract=True)
                else:
                    self._addCustomBondForce(near_potential, 1, force)
                    self._addCustomBondForce(minus_near_potential, 1, force)

    def _addCustomNonbondedForce(self, expressions, rcut, group, nonbonded):
        energy = ';'.join(expressions)
        force = atomsmm.forces._AtomsMM_CustomNonbondedForce(energy, rcut,
                                                             use_switching_function=False,
                                                             use_dispersion_correction=False)
        force.importFrom(nonbonded)
        force.setForceGroup(group)
        self.addForce(force)

    def _addCustomBondForce(self, expressions, group, nonbonded, extract=False):
        energy = ';'.join(expressions)
        force = atomsmm.forces._AtomsMM_CustomBondForce(energy)
        force.importFrom(nonbonded, extract)
        if force.getNumBonds() > 0:
            force.setForceGroup(group)
            self.addForce(force)


class SolvationSystem(openmm.System):
    """
    An OpenMM System_ prepared for solvation free-energy calculations.

    Parameters
    ----------
        system : openmm.System
            The original system from which to generate the SolvationSystem.
        solute_atoms : set(int)
            A set containing the indexes of all solute atoms.
        softcore_group : int, optional, default=0
            The force group to be assigned to the solute-solvent softcore interactions.
        split_exceptions : bool, optional, default=False
            Whether preexisting exceptions should be separated from the nonbonded force before new
            exceptions are created.

    """
    def __init__(self, system, solute_atoms, softcore_group=0, split_exceptions=False):
        self.this = copy.deepcopy(system).this
        nonbonded = self.getForce(atomsmm.findNonbondedForce(self))
        rcut = nonbonded.getCutoffDistance()
        rswitch = nonbonded.getSwitchingDistance() if nonbonded.getUseSwitchingFunction() else None
        all_atoms = set(range(nonbonded.getNumParticles()))
        solvent_atoms = all_atoms - solute_atoms

        # A custom nonbonded force for solute-solvent, softcore van der Waals interactions:
        softcore = atomsmm.SoftcoreLennardJonesForce(rcut, rswitch, parameter='lambda_vdw')
        softcore.importFrom(nonbonded)
        softcore.addInteractionGroup(solute_atoms, solvent_atoms)
        softcore.setForceGroup(softcore_group)
        self.addForce(softcore)

        # If requested, extract preexisting non-exclusion exceptions:
        if split_exceptions:
            ljc_potential = '4*epsilon*x*(x-1) + Kc*chargeprod/r; x=(sigma/r)^6; Kc=138.935456'
            exceptions = atomsmm.forces._AtomsMM_CustomBondForce(ljc_potential)
            exceptions.importFrom(nonbonded, extract=True)
            if exceptions.getNumBonds() > 0:
                self.addForce(exceptions)

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
                virial = atomsmm.forces._AtomsMM_CustomNonbondedForce(expression, rcut, True, rswitch)
                virial.importFrom(nonbonded)
                virial.setForceGroup(dispersionGroup)
                self.addForce(virial)
                exceptions = atomsmm.forces._AtomsMM_CustomBondForce(expression)
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
