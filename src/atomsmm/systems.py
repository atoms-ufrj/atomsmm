"""
.. module:: system
   :platform: Unix, Windows
   :synopsis: a module for defining extensions of OpenMM System_ class.

.. moduleauthor:: Charlles R. A. Abreu <abreu@eq.ufrj.br>

.. _System: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.System.html

"""

import copy
import itertools
import re

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
        self._special_bond_force = None
        self._special_angle_force = None
        adjustment = kwargs.pop('adjustment', 'force-switch')
        fastExceptions = kwargs.get('fastExceptions', True)
        ljc_potential = ['4*epsilon*x*(x-1) + Kc*chargeprod/r', 'x=(sigma/r)^6', 'Kc=138.935456']
        for force in self.getForces():
            if isinstance(force, openmm.NonbondedForce):
                near_potential = atomsmm.forces.nearForceExpressions(rcutIn, rswitchIn, adjustment)
                minus_near_potential = copy.deepcopy(near_potential)
                minus_near_potential[0] = '-step(rc0-r)*({})'.format(near_potential[0])
                force.setForceGroup(2)
                force.setReciprocalSpaceForceGroup(2)
                self._addCustomNonbondedForce(near_potential, rcutIn, 1, force)
                self._addCustomNonbondedForce(minus_near_potential, rcutIn, 31, force)
                if fastExceptions:
                    self._addCustomBondForce(ljc_potential, 0, force, extract=True)
                else:
                    self._addCustomBondForce(near_potential, 1, force)
                    self._addCustomBondForce(minus_near_potential, 31, force)
            elif isinstance(force, openmm.CustomNonbondedForce):
                potential = force.getEnergyFunction().split(';')
                if potential[0] in ['U_linear', 'U_spline', 'U_art', 'U_general']:
                    force.setForceGroup(2)
                    near_potential = atomsmm.forces.nearLJForceExpressions(rcutIn, rswitchIn, adjustment)
                    near_potential[0] = '((gt0-gt1)*S + gt1)*({})'.format(near_potential[0])
                    near_potential += ['gt0 = step(lambda_vdw)', 'gt1 = step(lambda_vdw-1)']
                    while not potential[0].startswith(' S ='):
                        potential.pop(0)
                    near_potential += potential
                    self._addCustomNonbondedForce(near_potential, rcutIn, 1, force)
                    near_potential[0] = '-step(rc0-r)*{}'.format(near_potential[0])
                    self._addCustomNonbondedForce(near_potential, rcutIn, 31, force)

    def _addCustomNonbondedForce(self, expressions, rcut, group, source):
        energy = ';'.join(expressions)
        if isinstance(source, openmm.NonbondedForce):
            force = atomsmm.forces._AtomsMM_CustomNonbondedForce(
                energy,
                rcut,
                use_switching_function=False,
                use_dispersion_correction=False,
            )
            force.importFrom(source)
        else:
            force = copy.deepcopy(source)
            force.setEnergyFunction(energy)
        force.setForceGroup(group)
        self.addForce(force)

    def _addCustomBondForce(self, expressions, group, nonbonded, extract=False):
        energy = ';'.join(expressions)
        force = atomsmm.forces._AtomsMM_CustomBondForce(energy)
        force.importFrom(nonbonded, extract)
        if force.getNumBonds() > 0:
            force.setForceGroup(group)
            self.addForce(force)

    def redefine_bond(self, topology, residue, atom1, atom2, length, K=None, group=1):
        """
        Changes the equilibrium length of a specified bond for integration within its original
        time scale. The difference between the original and the redefined bond potentials is
        evaluated at another time scale.

        Parameters
        ----------
            topology : openmm.Topology
                The topology corresponding to the original system.
            residue : str
                A name or regular expression to identify the residue which contains the redefined
                bond.
            atom1 : str
                A name or regular expression to identify the first atom that makes the bond.
            atom2 : str
                A name or regular expression to identify the second atom that makes the bond.
            length : unit.Quantity
                The redifined equilibrium length for integration at the shortest time scale.
            K : unit.Quantity, optional, default=None
                The harmonic force constant for the bond. If this is `None`, then the original
                value will be maintained.
            group : int, optional, default=1
                The force group with which the difference between the original and the redefined
                bond potentials must be evaluated.

        """
        resname = [atom.residue.name for atom in topology.atoms()]
        atom = [atom.name for atom in topology.atoms()]
        r_regex = re.compile(residue)
        a_regex = [re.compile(a) for a in [atom1, atom2]]

        def r_match(*args):
            return all(r_regex.match(resname[j]) for j in args)

        def a_match(*args):
            return all(a_regex[i].match(atom[j]) for i, j in enumerate(args))

        bond_list = []
        for force in self.getForces():
            if isinstance(force, openmm.HarmonicBondForce):
                for index in range(force.getNumBonds()):
                    i, j, r0, K0 = force.getBondParameters(index)
                    if r_match(i, j) and (a_match(i, j) or a_match(j, i)):
                        force.setBondParameters(index, i, j, length, K0 if K is None else K)
                        bond_list.append((i, j, r0, K0))
        if bond_list and self._special_bond_force is None:
            new_force = openmm.CustomBondForce('0.5*(K0*(r - r0)^2 - Kn*(r - rn)^2)')
            new_force.addPerBondParameter('r0')
            new_force.addPerBondParameter('K0')
            new_force.addPerBondParameter('rn')
            new_force.addPerBondParameter('Kn')
            new_force.setForceGroup(group)
            self.addForce(new_force)
            self._special_bond_force = new_force
        for (i, j, r0, K0) in bond_list:
            self._special_bond_force.addBond(i, j, (r0, K0, length, K0 if K is None else K))

    def redefine_angle(self, topology, residue, atom1, atom2, atom3, angle, K=None, group=1):
        """
        Changes the equilibrium value of a specified angle for integration within its original
        time scale. The difference between the original and the redefined angle potentials is
        evaluated at another time scale.

        Parameters
        ----------
            topology : openmm.Topology
                The topology corresponding to the original system.
            residue : str
                A name or regular expression to identify the residue which contains the redefined
                angle.
            atom1 : str
                A name or regular expression to identify the first atom that makes the angle.
            atom2 : str
                A name or regular expression to identify the second atom that makes the angle.
            atom3 : str
                A name or regular expression to identify the third atom that makes the angle.
            angle : unit.Quantity
                The redifined equilibrium angle value for integration at the shortest time scale.
            K : unit.Quantity, optional, default=None
                The harmonic force constant for the angle. If this is `None`, then the original
                value will be maintained.
            group : int, optional, default=1
                The force group with which the difference between the original and the redefined
                angle potentials must be evaluated.

        """
        resname = [atom.residue.name for atom in topology.atoms()]
        atom = [atom.name for atom in topology.atoms()]
        r_regex = re.compile(residue)
        a_regex = [re.compile(a) for a in [atom1, atom2, atom3]]

        def r_match(*args):
            return all(r_regex.match(resname[j]) for j in args)

        def a_match(*args):
            return all(a_regex[i].match(atom[j]) for i, j in enumerate(args))

        angle_list = []
        for force in self.getForces():
            if isinstance(force, openmm.HarmonicAngleForce):
                for index in range(force.getNumAngles()):
                    i, j, k, theta0, K0 = force.getAngleParameters(index)
                    if r_match(i, j, k) and (a_match(i, j, k) or a_match(k, j, i)):
                        force.setAngleParameters(index, i, j, k, angle, K0 if K is None else K)
                        angle_list.append((i, j, k, theta0, K0))
        if angle_list and self._special_angle_force is None:
            new_force = openmm.CustomAngleForce('0.5*(K0*(theta - t0)^2 - Kn*(theta - tn)^2)')
            new_force.addPerAngleParameter('t0')
            new_force.addPerAngleParameter('K0')
            new_force.addPerAngleParameter('tn')
            new_force.addPerAngleParameter('Kn')
            new_force.setForceGroup(group)
            self.addForce(new_force)
            self._special_angle_force = new_force
        for (i, j, k, theta0, K0) in angle_list:
            self._special_angle_force.addAngle(i, j, k, (theta0, K0, angle, K0 if K is None else K))


class SolvationSystem(openmm.System):
    """
    An OpenMM System_ prepared for solvation free-energy calculations.

    Parameters
    ----------
        system : openmm.System
            The original system from which to generate the SolvationSystem.
        solute_atoms : set(int)
            A set containing the indexes of all solute atoms.
        use_softcore : bool, optional, default=True
            Whether to define a softcore potential for the coupling/decoupling of solute-solvent
            Lennard-Jones interactions. If this is `False`, then a linear scaling of both `sigma`
            and `epsilon` will be applied instead.
        softcore_group : int, optional, default=0
            The force group to be assigned to the solute-solvent softcore interactions, if any.
        split_exceptions : bool, optional, default=False
            Whether preexisting exceptions should be separated from the nonbonded force before new
            exceptions are created.

    """
    def __init__(self, system, solute_atoms, use_softcore=True, softcore_group=0, split_exceptions=False):
        self.this = copy.deepcopy(system).this
        nonbonded = self.getForce(atomsmm.findNonbondedForce(self))
        all_atoms = set(range(nonbonded.getNumParticles()))
        solvent_atoms = all_atoms - solute_atoms

        # If requested, extract preexisting non-exclusion exceptions:
        if split_exceptions:
            ljc_potential = '4*epsilon*x*(x-1) + Kc*chargeprod/r; x=(sigma/r)^6; Kc=138.935456'
            exceptions = atomsmm.forces._AtomsMM_CustomBondForce(ljc_potential)
            exceptions.importFrom(nonbonded, extract=True)
            if exceptions.getNumBonds() > 0:
                self.addForce(exceptions)

        # A custom nonbonded force for solute-solvent, softcore van der Waals interactions:
        if use_softcore:
            ljs_potential = '4*lambda_vdw*epsilon*(1-x)/x^2; x=(r/sigma)^6+0.5*(1-lambda_vdw)'
            softcore = atomsmm.forces._AtomsMM_CustomNonbondedForce(ljs_potential, lambda_vdw=1)
            softcore.importFrom(nonbonded)
            softcore.addInteractionGroup(solute_atoms, solvent_atoms)
            softcore.setForceGroup(softcore_group)
            self.addForce(softcore)

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
                if use_softcore:
                    softcore.addExclusion(i, j)  # Needed for matching exception number

        # Turn off or scale solute Lennard-Jones interactions, scale solute charges:
        lj_parameters = dict()
        charges = dict()
        for index in solute_atoms:
            charge, sigma, epsilon = nonbonded.getParticleParameters(index)
            nonbonded.setParticleParameters(index, 0.0, 0.0, 0.0)
            if charge/charge.unit != 0.0:
                charges[index] = charge
            if epsilon/epsilon.unit != 0.0:
                lj_parameters[index] = (sigma, epsilon)
        if charges:
            nonbonded.addGlobalParameter('lambda_coul', 1.0)
            for index, charge in charges.items():
                nonbonded.addParticleParameterOffset('lambda_coul', index, charge, 0.0, 0.0)
        if lj_parameters and not use_softcore:
            nonbonded.addGlobalParameter('lambda_vdw', 1.0)
            for index, (sigma, epsilon) in lj_parameters.items():
                nonbonded.addParticleParameterOffset('lambda_vdw', index, 0.0, sigma, epsilon)


class AlchemicalSystem(openmm.System):
    """
    An OpenMM System_ prepared for solvation free-energy calculations.

    Parameters
    ----------
        system : openmm.System
            The original system from which to generate the SolvationSystem.
        atoms : set(int)
            A set containing the indexes of all solute atoms.
        coupling : str, optional, default='softcore'
            The model used for coupling the alchemical atoms to the system. The options are
            `softcore`, `linear`, `art`, `spline`, or some function of `lambda_vdw`. Use `softcore`
            for the model of Beutler et al. (1994), `linear` for a simple linear coupling, `art` for
            the sine-based coupling model of Abrams, Rosso, and Tuckerman (2006), and `spline` for
            multiplying the solute-solvent interactions by
            :math:`\\lambda_\\mathrm{vdw}^3(10 - 15 \\lambda_\\mathrm{vdw} + 6 \\lambda_\\mathrm{vdw}^2)`.
            Alternatively, you can enter any other valid function of `lambda_vdw`.
        group : int, optional, default=0
            The force group to be assigned to the solute-solvent softcore interactions, if any.
        use_lrc : bool, optional, defaul=False
            Whether to use long-range (dispersion) correction in solute-solvent interactions.

    """
    def __init__(self, system, atoms, coupling='softcore', group=0, use_lrc=False):
        self.this = copy.deepcopy(system).this
        nonbonded = self.getForce(atomsmm.findNonbondedForce(self))

        if coupling == 'softcore':  # Beutler et al. (1994)
            potential = 'U_softcore'
            potential += '; U_softcore = 4*lambda_vdw*epsilon*(1 - x)/x^2'
            potential += '; x = (r/sigma)^6 + 0.5*(1 - lambda_vdw)'
        else:
            if coupling in ['linear', 'spline', 'art']:
                potential = 'U_{}'.format(coupling)
            else:
                potential = 'U_general'
            potential += '; {} = 4*((gt0-gt1)*S + gt1)*epsilon*x*(x - 1)'.format(potential)
            potential += '; x = (sigma/r)^6'
            potential += '; gt0 = step(lambda_vdw)'
            potential += '; gt1 = step(lambda_vdw-1)'
            if coupling == 'linear':
                potential += '; S = lambda_vdw - sin(two_pi*lambda_vdw)/two_pi'
            elif coupling == 'spline':
                potential += '; S = lambda_vdw^3*(10 - 15*lambda_vdw + 6*lambda_vdw^2)'
            elif coupling == 'art':  # Abrams, Rosso, and Tuckerman (2006)
                potential += '; S = lambda_vdw - sin(two_pi*lambda_vdw)/two_pi'
                potential += '; two_pi = 6.28318530717958'
            else:
                potential += '; S = {}'.format(coupling)
        potential += '; sigma = 0.5*(sigma1 + sigma2)'
        potential += '; epsilon = sqrt(epsilon1*epsilon2)'
        softcore = openmm.CustomNonbondedForce(potential)
        if nonbonded.getNonbondedMethod() == openmm.NonbondedForce.NoCutoff:
            softcore.setNonbondedMethod(openmm.CustomNonbondedForce.NoCutoff)
        else:
            softcore.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
        softcore.setCutoffDistance(nonbonded.getCutoffDistance())
        softcore.setUseSwitchingFunction(nonbonded.getUseSwitchingFunction())
        softcore.setSwitchingDistance(nonbonded.getSwitchingDistance())
        # softcore.setUseLongRangeCorrection(nonbonded.getUseDispersionCorrection())
        softcore.setUseLongRangeCorrection(use_lrc)
        softcore.addGlobalParameter('lambda_vdw', 1.0)
        softcore.addPerParticleParameter('sigma')
        softcore.addPerParticleParameter('epsilon')
        all_atoms = range(nonbonded.getNumParticles())
        for index in all_atoms:
            _, sigma, epsilon = nonbonded.getParticleParameters(index)
            softcore.addParticle([sigma, epsilon])
        for index in range(nonbonded.getNumExceptions()):
            i, j, _, _, epsilon = nonbonded.getExceptionParameters(index)
            softcore.addExclusion(i, j)
        softcore.addInteractionGroup(atoms, set(all_atoms) - set(atoms))
        softcore.setForceGroup(group)
        softcore.addEnergyParameterDerivative('lambda_vdw')
        self.addForce(softcore)

        parameters = []
        for index in atoms:
            parameters.append(nonbonded.getParticleParameters(index))
            nonbonded.setParticleParameters(index, 0.0, 1.0, 0.0)

        exception_pairs = []
        for index in range(nonbonded.getNumExceptions()):
            i, j, _, _, _ = nonbonded.getExceptionParameters(index)
            if set([i, j]).issubset(atoms):
                exception_pairs.append(set([i, j]))
        for i, j in itertools.combinations(atoms, 2):
            if set([i, j]) not in exception_pairs:
                q1, sig1, eps1 = parameters[i]
                q2, sig2, eps2 = parameters[j]
                nonbonded.addException(i, j, q1*q2, (sig1 + sig2)/2, np.sqrt(eps1*eps2))
                softcore.addExclusion(i, j)  # Needed for matching exception number


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
    def __init__(self, system):
        super().__init__(system, copyForces=False)
        dispersionGroup = 0
        bondedGroup = 1
        coulombGroup = 2
        self._dispersion = 2**dispersionGroup
        self._bonded = 2**bondedGroup
        self._coulomb = 2**coulombGroup
        for force in system.getForces():
            if isinstance(force, openmm.NonbondedForce) and force.getNumParticles() > 0:
                nonbonded = copy.deepcopy(force)
                expression = '24*epsilon*(2*(sigma/r)^12-(sigma/r)^6)'
                virial = atomsmm.forces._AtomsMM_CustomNonbondedForce(expression)
                virial.importFrom(nonbonded)
                virial.setForceGroup(dispersionGroup)
                self.addForce(virial)
                exceptions = atomsmm.forces._AtomsMM_CustomBondForce(expression)
                exceptions.importFrom(nonbonded, extract=False)
                if exceptions.getNumBonds() > 0:
                    exceptions.setForceGroup(dispersionGroup)
                    self.addForce(exceptions)
                for index in range(nonbonded.getNumParticles()):
                    charge, _, _ = nonbonded.getParticleParameters(index)
                    nonbonded.setParticleParameters(index, charge, 1.0, 0.0)
                for index in range(nonbonded.getNumExceptions()):
                    i, j, charge, _, _ = nonbonded.getExceptionParameters(index)
                    nonbonded.setExceptionParameters(index, i, j, charge, 1.0, 0.0)
                nonbonded.setForceGroup(coulombGroup)
                nonbonded.setReciprocalSpaceForceGroup(coulombGroup)
                self.addForce(nonbonded)
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
