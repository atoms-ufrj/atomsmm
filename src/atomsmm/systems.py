"""
.. module:: system
   :platform: Unix, Windows
   :synopsis: a module for defining extensions of OpenMM System_ class.

.. moduleauthor:: Charlles R. A. Abreu <abreu@eq.ufrj.br>

.. _System: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.System.html
.. _Context: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.Context.html

"""

import copy
import itertools
import re

import numpy as np
from simtk import openmm
from simtk import unit
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

        parameters = {}
        for i in atoms:
            parameters[i] = nonbonded.getParticleParameters(i)
            nonbonded.setParticleParameters(i, 0.0, 1.0, 0.0)

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


class AlchemicalSoftcoreCVForce(object):
    def __init__(self, alchemical_system, grid):
        self._system = openmm.System()
        for i in range(alchemical_system.getNumParticles()):
            self._system.addParticle(alchemical_system.getParticleMass(i))
        self._system.setDefaultPeriodicBoxVectors(*alchemical_system.getDefaultPeriodicBoxVectors())
        self._context = None
        self._numForces = len(grid)
        original = alchemical_system._alchemical_vdw_force
        # nonbonded = openmm.NonbondedForce()
        # for i in range(original.getNumParticles()):
        #     nonbonded.addParticle(*original.getParticleParameters(i))
        # nonbonded.setForceGroup(31)
        # self._system.addForce(nonbonded)  # For keeping neighbor list
        for index, value in enumerate(grid):
            ljsoft = f'4*lambda*epsilon*x*(x - 1)'
            ljsoft += f'; x = 1/((r/sigma)^6 + 0.5*(1-lambda))'
            ljsoft += f'; lambda = {value}'
            ljsoft += f'; sigma = 0.5*(sigma1 + sigma2)'
            ljsoft += f'; epsilon = sqrt(epsilon1*epsilon2)'
            force = openmm.CustomNonbondedForce(ljsoft)
            force.setNonbondedMethod(original.getNonbondedMethod())
            for parameter in ['sigma', 'epsilon']:
                force.addPerParticleParameter(parameter)
            for i in range(original.getNumParticles()):
                _, sigma, epsilon = original.getParticleParameters(i)
                force.addParticle((sigma, epsilon))
            for i in range(original.getNumExclusions()):
                force.addExclusion(*original.getExclusionParticles(i))
            force.setCutoffDistance(original.getCutoffDistance())
            force.setUseSwitchingFunction(original.getUseSwitchingFunction())
            force.setSwitchingDistance(original.getSwitchingDistance())
            if value != 0.0:
                force.setUseLongRangeCorrection(original.getUseLongRangeCorrection())
            for i in range(original.getNumInteractionGroups()):
                force.addInteractionGroup(*original.getInteractionGroupParameters(i))
            force.setForceGroup(index)
            self._system.addForce(force)

    def getNumCollectiveVariables(self):
        return self._numForces

    def getCollectiveVariableName(self, index):
        return f'E{index}'

    def getCollectiveVariableValues(self, context):
        if self._context is None:
            integrator = openmm.CustomIntegrator(0)
            platform = context.getPlatform()
            self._context = openmm.Context(self._system, integrator, platform)
        self._context.setState(context.getState(getPositions=True))
        energy = np.empty(self._numForces)
        for index in range(self._numForces):
            state = self._context.getState(getEnergy=True, groups=set([index]))
            energy[index] = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        return energy


class AlchemicalCoulombCVForce(object):
    def __init__(self, alchemical_system):
        self._system = alchemical_system

    def getNumCollectiveVariables(self):
        return 1

    def getCollectiveVariableName(self, index):
        return 'alchemical_coulomb_energy'

    def getCollectiveVariableValues(self, context):
        lambda_coul = self._system._lambda_coul
        self._system.reset_coulomb_scaling_factor(0.0, context)
        group = 2 if self._system._middle_scale else 1
        E0 = context.getState(getEnergy=True, groups=2**group).getPotentialEnergy()
        self._system.reset_coulomb_scaling_factor(1.0, context)
        E1 = context.getState(getEnergy=True, groups=2**group).getPotentialEnergy()
        self._system.reset_coulomb_scaling_factor(lambda_coul, context)
        return [(E1 - E0).value_in_unit(unit.kilojoules_per_mole)]


class AlchemicalRespaSystem(openmm.System):
    """
    An OpenMM System_ prepared for Multiple Time-Scale Integration with RESPA and for alchemical
    coupling/decoupling of specified atoms.

    Short-range forces for integration at intermediate time scale are generated by applying a
    switching function to the force that results from the following potential:

    .. math::
        & V(r)=V^\\ast_\\mathrm{LJC}(r)-V^\\ast_\\mathrm{LJC}(r_\\mathrm{cut,in}) \\\\
        & V^\\ast_\\mathrm{LJC}(r)=\\left\\{
            4\\epsilon\\left[f_{12}(u(r))\\left(\\frac{\\sigma}{r}\\right)^{12}-f_6(u(r))\\left(\\frac{\\sigma}{r}\\right)^6\\right]
            + \\frac{f_1(u(r))}{4\\pi\\epsilon_0}\\frac{q_1 q_2}{r}
        \\right\\}

    where :math:`f_n(u)` is the solution of the 1st order differential equation

    .. math::
        & f_n-\\frac{u+b}{n}\\frac{df_n}{du}=S(u) \\\\
        & f_n(0)=1 \\\\
        & b=\\frac{r_\\mathrm{switch,in}}{r_\\mathrm{cut,in}-r_\\mathrm{switch,in}}

    As a consequence of this modification, :math:`V^\\prime(r)=S(u(r))V^\\prime_\\mathrm{LJC}(r)`.

    Examples of coupling function are:

    1. Linear coupling (default):

    .. math::
        f(\\lambda) = \\lambda

    2. A 5-th order polinomial whose 1st- and 2nd-order derivatives are null at both extremes.

    .. math::
        f(\\lambda) = \\lambda^3(10 - 15 \\lambda + 6 \\lambda^2)

    3. A 5-th order polinomial whose 1st-order derivative is null at both extremes and whose 2nd-
    and 3rd-order derivatives are also null at :math:`\\lambda = 0`:

    .. math::
        f(\\lambda) = \\lambda^4(5 - 4 \\lambda)

    4. The sine-based coupling model of Abrams, Rosso, and Tuckerman :cite:`Abrams_2006`:

    .. math::
        f(\\lambda) = \\lambda - \\frac{\\sin\\left(2\\pi \\lambda\\right)}{2\\pi}

    Parameters
    ----------
        system : openmm.System
            The original system from which to generate the SolvationSystem.
        rcutIn : unit.Quantity
            The distance at which the short-range nonbonded interactions will completely vanish.
        rswitchIn : unit.Quantity
            The distance at which the short-range nonbonded interactions will start vanishing by
            application of a switching function.
        alchemical_atoms : list(int), optional, default=[]
            A set containing the indexes of all alchemical atoms.
        coupling_parameter : str, optional, defaul='lambda'
            The name of the coupling parameter.
        coupling_function : str, optional, default='lambda'
            A function :math:`f(\\lambda)` used for coupling the alchemical atoms to the system,
            where :math:`\\lambda` is the coupling parameter. This must be a function of a single
            variable named as in argument `coupling_parameter` (see above). It is expected that
            :math:`f(0) = 0` and :math:`f(1) = 1`.
        middle_scale : bool, optional, default=True
            Whether to use an intermediate time scale in the RESPA integration.
        coulomb_scaling : bool, optional, default=False
            Whether to consider scaling of electrostatic interactions between alchemical and
            non-alchemical atoms. Otherwise, these interactions will not exist.
        lambda_coul : float, optional, default=0
            A scaling factor to be applied to all electrostatic interactions between alchemical and
            non-alchemical atoms.

    """
    def __init__(self, system, rcutIn, rswitchIn, alchemical_atoms=[],
                 coupling_parameter='lambda', coupling_function='lambda',
                 middle_scale=True, coulomb_scaling=False, lambda_coul=0,
                 use_softcore=False):
        self.this = copy.deepcopy(system).this
        Kc = 138.935456637  # Coulomb constant in kJ.nm/mol.e^2

        self._parameter = coupling_parameter
        self._coulomb_scaling = coulomb_scaling
        self._middle_scale = middle_scale
        self._use_softcore = use_softcore

        # Define specific sets of atoms:
        all_atoms = set(range(self.getNumParticles()))
        solute_atoms = set(alchemical_atoms)
        solvent_atoms = all_atoms - solute_atoms

        # Define force-switched potential expressions:
        rci = rcutIn.value_in_unit(unit.nanometer)
        rsi = rswitchIn.value_in_unit(unit.nanometer)
        fsp = self._force_switched_potential(rci, rsi, Kc)

        mixing_rules = '; chargeprod = charge1*charge2'
        mixing_rules += '; sigma = 0.5*(sigma1 + sigma2)'
        mixing_rules += '; epsilon = sqrt(epsilon1*epsilon2)'

        # Nonbonded force will only account for solvent-solvent interactions:
        for force in self.getForces():
            if isinstance(force, openmm.NonbondedForce):
                # Store a copy of the nonbonded force before changes are made:
                nonbonded = copy.deepcopy(force)

                # Place it at due group, delete all solute interaction parameters:
                force.setForceGroup(2 if middle_scale else 1)
                force.setReciprocalSpaceForceGroup(2 if middle_scale else 1)
                self._solute_charges = {}
                for i in solute_atoms:
                    charge, _, _ = force.getParticleParameters(i)
                    self._solute_charges[i] = charge
                    force.setParticleParameters(i, 0.0, 1.0, 0.0)

                # Identify solute-solute exceptions and turn all of them into exclusions:
                exception_pairs = []
                for index in range(force.getNumExceptions()):
                    i, j, _, _, _ = nonbonded.getExceptionParameters(index)
                    if set([i, j]).issubset(solute_atoms):
                        exception_pairs.append(set([i, j]))
                        force.setExceptionParameters(index, i, j, 0.0, 1.0, 0.0)

                # Identify all other solute-solute interactions. In the system's nonbonded force,
                # turn them into exclusion exceptions. In the stored copy, turn them into general
                # exceptions for the sake of forthcoming imports:
                for i, j in itertools.combinations(solute_atoms, 2):
                    if set([i, j]) not in exception_pairs:
                        force.addException(i, j, 0.0, 1.0, 0.0)
                        q1, sig1, eps1 = nonbonded.getParticleParameters(i)
                        q2, sig2, eps2 = nonbonded.getParticleParameters(j)
                        nonbonded.addException(i, j, q1*q2, (sig1 + sig2)/2, np.sqrt(eps1*eps2))

                # Add a force-switched potential with internal cutoff if a RESPA-related
                # middle scale has been requested:
                if middle_scale:
                    near_force = openmm.CustomNonbondedForce(fsp + mixing_rules)
                    self._import_from_nonbonded(near_force, force)
                    near_force.setCutoffDistance(rcutIn)
                    near_force.setUseSwitchingFunction(False)
                    near_force.setUseLongRangeCorrection(False)
                    near_force.addGlobalParameter('respa_switch', 0)
                    near_force.setForceGroup(1)
                    self.addForce(near_force)

                    # Because all exceptions in nonbonded become exclusions in near_force,
                    # capture all non-exclusion exceptions into a custom bonded force:
                    exceptions = openmm.CustomBondForce(f'step({rci}-r)*U; U = {fsp}')
                    exceptions.addGlobalParameter('respa_switch', 0)
                    for parameter in ['chargeprod', 'sigma', 'epsilon']:
                        exceptions.addPerBondParameter(parameter)
                    for index in range(force.getNumExceptions()):
                        i, j, chargeprod, sigma, epsilon = force.getExceptionParameters(index)
                        if chargeprod/chargeprod.unit != 0.0 or epsilon/epsilon.unit != 0.0:
                            exceptions.addBond(i, j, (chargeprod, sigma, epsilon))
                    if exceptions.getNumBonds() > 0:
                        exceptions.setForceGroup(1)
                        self.addForce(exceptions)

                # Store a pointer to the altered non-bonded force:
                self._nonbonded_force = force
            else:
                # Place bonded and other forces at group 0:
                force.setForceGroup(0)

        # To allow decoupling rather than annihilation, solute-solute interactions are handled
        # by a custom bond force without cut-off:
        ljc = f'4*epsilon*x*(x - 1) + {Kc}*chargeprod/r; x = (sigma/r)^6'
        full_range = openmm.CustomBondForce(ljc)
        full_range.setForceGroup(2 if middle_scale else 1)
        intrasolute_forces = [full_range]

        # If a RESPA-related middle scale has been requested, also create a short-ranged version:
        if middle_scale:
            short_range = openmm.CustomBondForce(f'step({rci}-r)*U; U = {fsp}')
            short_range.addGlobalParameter('respa_switch', 0)
            short_range.setForceGroup(1)
            intrasolute_forces.append(short_range)

        for force in intrasolute_forces:
            for parameter in ['chargeprod', 'sigma', 'epsilon']:
                force.addPerBondParameter(parameter)
            self.addForce(force)

        # Add interactions due to solute-solute pairs previously treated as exceptions:
        for index in range(nonbonded.getNumExceptions()):
            i, j, chargeprod, sigma, epsilon = nonbonded.getExceptionParameters(index)
            if set([i, j]).issubset(solute_atoms):
                for force in intrasolute_forces:
                    force.addBond(i, j, (chargeprod, sigma, epsilon))

        # NOTE: if Coulomb scaling treatment was requested, the electrostatic part of full-ranged
        # solute-solvent interactions will be enabled by reactivating solute charges while keeping
        # all intra-solute interactions excluded.

        # If both Coulomb scaling and a RESPA-related middle scale were requested, the electrostatic
        # part of short-ranged solute-solvent interactions must be added as well:
        if coulomb_scaling and middle_scale:
            # Create a force-switched electrostatic potential and add it to the system:
            fsep = self._force_switched_eletrostatic_potential(rci, rsi, Kc)
            short_range = openmm.CustomNonbondedForce(fsep + mixing_rules)
            self._import_from_nonbonded(short_range, nonbonded)
            short_range.setCutoffDistance(rcutIn)
            short_range.setUseSwitchingFunction(False)
            short_range.setUseLongRangeCorrection(False)
            short_range.addGlobalParameter('respa_switch', 0)
            short_range.setForceGroup(1)
            short_range.addInteractionGroup(solute_atoms, solvent_atoms)
            self.addForce(short_range)
            self._fsep_force = short_range

        if use_softcore:
            # Softcore potential is fully considered in the middle time scale:
            ljsoft = f'4*{coupling_parameter}*epsilon*x*(x - 1)'
            ljsoft += f'; x = 1/((r/sigma)^6 + 0.5*(1-{coupling_parameter}))'
            full_range = openmm.CustomNonbondedForce(ljsoft + mixing_rules)
            self._import_from_nonbonded(full_range, nonbonded, import_globals=True)
            full_range.addInteractionGroup(solute_atoms, solvent_atoms)
            full_range.addGlobalParameter(coupling_parameter, 1.0)
            full_range.addEnergyParameterDerivative(coupling_parameter)
            full_range.setForceGroup(2 if middle_scale else 1)
            self.addForce(full_range)

            # Store force object related to alchemical coupling/decoupling:
            self._alchemical_vdw_force = full_range

            if middle_scale:
                # In the current version, the full softcore potential is allocated in the middle
                # time scale because it would be difficult to apply the force-switch strategy. This
                # might be reviewed in the future.
                short_range = copy.deepcopy(full_range)
                short_range.setEnergyFunction(f'respa_switch*{ljsoft}' + mixing_rules)
                short_range.addGlobalParameter('respa_switch', 0)
                short_range.setForceGroup(1)
                self.addForce(short_range)

        else:
            # The van der Waals part of solute-solvent interactions are defined as collective
            # variables multiplied by a coupling function:
            potential = '((gt0-gt1)*S + gt1)*alchemical_vdw_energy'
            potential += f'; gt0 = step({coupling_parameter})'
            potential += f'; gt1 = step({coupling_parameter}-1)'
            potential += f'; S = {coupling_function}'
            cv_force = openmm.CustomCVForce(potential)
            cv_force.addGlobalParameter(coupling_parameter, 1.0)
            cv_force.addEnergyParameterDerivative(coupling_parameter)

            # For the van der Waals part of solute-solvent interactions, it is considered that no
            # exceptions exist which involve a solute atom and a solvent atom (this might be
            # reviewed in future versions):
            lj = f'4*epsilon*x*(x - 1); x = (sigma/r)^6'
            full_range = openmm.CustomNonbondedForce(lj + mixing_rules)
            self._import_from_nonbonded(full_range, nonbonded, import_globals=True)
            full_range.addInteractionGroup(solute_atoms, solvent_atoms)
            full_range_cv_force = copy.deepcopy(cv_force)
            full_range_cv_force.addCollectiveVariable('alchemical_vdw_energy', full_range)
            full_range_cv_force.setForceGroup(2 if middle_scale else 1)
            self.addForce(full_range_cv_force)

            # Store force object related to alchemical coupling/decoupling:
            self._alchemical_vdw_force = full_range_cv_force

            if middle_scale:
                fsljp = self._force_switched_potential(rci, rsi, 0.0)
                short_range = openmm.CustomNonbondedForce(fsljp + mixing_rules)
                self._import_from_nonbonded(short_range, nonbonded)
                short_range.setCutoffDistance(rcutIn)
                short_range.setUseSwitchingFunction(False)
                short_range.setUseLongRangeCorrection(False)
                short_range.addGlobalParameter('respa_switch', 0)
                short_range.addInteractionGroup(solute_atoms, solvent_atoms)
                short_range_cv_force = cv_force
                short_range_cv_force.addCollectiveVariable('alchemical_vdw_energy', short_range)
                short_range_cv_force.setForceGroup(1)
                self.addForce(short_range_cv_force)

        # Store Coulomb scaling constant as zero, but reset it if a different value has been passed:
        self._lambda_coul = 0
        self.reset_coulomb_scaling_factor(lambda_coul)

    def get_alchemical_vdw_force(self, parameter_values=[1]):
        if self._use_softcore:
            return AlchemicalSoftcoreCVForce(self, parameter_values)
        else:
            return self._alchemical_vdw_force

    def get_alchemical_coul_force(self):
        return AlchemicalCoulombCVForce(self)

    def reset_coulomb_scaling_factor(self, lambda_coul, context=None):
        """
        Resets the scaling factor of the solute-solvent electrostatic interactions.

        Parameters
        ----------
            lambda_coul : float
                The scaling factor value.
            context : Context_, optional, default=None
                A context in which the particle parameters should be updated.

        """
        if self._coulomb_scaling and lambda_coul != self._lambda_coul:
            for i, charge in self._solute_charges.items():
                self._nonbonded_force.setParticleParameters(i, lambda_coul*charge, 1.0, 0.0)
                if self._middle_scale:
                    self._fsep_force.setParticleParameters(i, (lambda_coul*charge, 1.0, 0.0))
            if context is not None:
                self._nonbonded_force.updateParametersInContext(context)
                if self._middle_scale:
                    self._fsep_force.updateParametersInContext(context)
            self._lambda_coul = lambda_coul

    def _force_switched_potential(self, rc, rs, Kc):
        b = rs/(rc - rs)
        a12 = (6*b**2-21*b+28)/462
        a6 = 6*b**2-3*b+1
        f = {}
        f[12] = f'{a12}*({b**3}*(R^12-1)-{12*b**2}*u-{66*b}*u^2-220*u^3)+({45*(7-2*b)/14})*u^4-{72/7}*u^5'
        f[6] = f'{a6}*({b**3}*(R^6-1)-{6*b**2}*u-{15*b}*u^2-20*u^3)+({45*(1-2*b)})*u^4-36*u^5'
        if Kc == 0.0:
            fsp = f'respa_switch*(4*epsilon*x*(x-1) + step(r-{rs})*perturbation)'
            fsp += f'; perturbation = 4*epsilon*x*(f12*x-f6)'
        else:
            a1 = 5*(b+1)**2
            f[1] = f'{a1}*({6*b**3}*R*log(R)-{6*b**2}*u-{3*b}*u^2+u^3)-{5*(b/2+1)}*u^4+{3/2}*u^5'
            fsp = f'respa_switch*(4*epsilon*x*(x-1) + {Kc}*chargeprod/r + step(r-{rs})*perturbation)'
            fsp += f'; perturbation = 4*epsilon*x*(f12*x-f6) + {Kc}*f1*chargeprod/r'
        fsp += '; x = (sigma/r)^6'
        for variable, expression in f.items():
            fsp += f'; f{variable} = {expression}'
        fsp += f'; R = {1/b}*u + 1'
        fsp += f'; u = {b/rs}*r - {b}'
        return fsp

    def _force_switched_eletrostatic_potential(self, rc, rs, Kc):
        b = rs/(rc - rs)
        a1 = 5*(b+1)**2
        f1 = f'{a1}*({6*b**3}*R*log(R)-{6*b**2}*u-{3*b}*u^2+u^3)-{5*(b/2+1)}*u^4+{3/2}*u^5'
        fsep = f'respa_switch*(1 + step(r-{rs})*f1)*{Kc}*chargeprod/r'
        fsep += f'; f1 = {f1}'
        fsep += f'; R = {1/b}*u + 1'
        fsep += f'; u = {b/rs}*r - {b}'
        return fsep

    def _import_from_nonbonded(self, force, nonbonded, import_globals=False):
        if nonbonded.getNonbondedMethod() == openmm.NonbondedForce.NoCutoff:
            force.setNonbondedMethod(openmm.CustomNonbondedForce.NoCutoff)
        else:
            force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
        for parameter in ['charge', 'sigma', 'epsilon']:
            force.addPerParticleParameter(parameter)
        for i in range(nonbonded.getNumParticles()):
            force.addParticle(nonbonded.getParticleParameters(i))
        for index in range(nonbonded.getNumExceptions()):
            i, j, _, _, _ = nonbonded.getExceptionParameters(index)
            force.addExclusion(i, j)
        if import_globals:
            force.setCutoffDistance(nonbonded.getCutoffDistance())
            force.setUseSwitchingFunction(nonbonded.getUseSwitchingFunction())
            force.setSwitchingDistance(nonbonded.getSwitchingDistance())
            force.setUseLongRangeCorrection(nonbonded.getUseDispersionCorrection())


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
