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
        fastExceptions : bool, optional, default=True
            Whether nonbonded exceptions must be considered to belong to the group of fastest
            forces.

    """
    def __init__(self, system, rcutIn, rswitchIn, **kwargs):
        fastExceptions = kwargs.get('fastExceptions', True)
        super().__init__(system)

        for force in self.getForces():
            if isinstance(force, openmm.NonbondedForce):
                force.setForceGroup(2)
                force.setReciprocalSpaceForceGroup(2)

                exceptions = atomsmm.NonbondedExceptionsForce()
                exceptions.extractFrom(force)
                if exceptions.getNumBonds() > 0:
                    exceptions.setForceGroup(0 if fastExceptions else 2)
                    self.addForce(exceptions)

                adjustment = kwargs.pop('adjustment', 'force-switch')
                innerForce = atomsmm.NearNonbondedForce(rcutIn, rswitchIn, adjustment)
                innerForce.importFrom(force)
                innerForce.setForceGroup(1)
                self.addForce(innerForce)

                potential = innerForce.getEnergyFunction().split(';')
                potential[0] = '-step(rc0-r)*({})'.format(potential[0])
                potential = ';'.join(potential)
                cutoff = force.getCutoffDistance()
                globals = innerForce.getGlobalParameters()
                discount = atomsmm.forces._AtomsMM_CustomNonbondedForce(potential, cutoff, None, **globals)
                discount.importFrom(force)
                discount.setForceGroup(2)
                self.addForce(discount)
            else:
                force.setForceGroup(0)


class SolvationSystem(_AtomsMM_System):
    """
    An OpenMM System_ prepared for solvation free-energy calculation.

    Parameters
    ----------
        system : openmm.System
            The original system from which to generate the SolvationSystem.
        solute_atoms : list
            A list containing the indexes of all solute atoms.
        forceGroup : int, optional, default=0
            The force group to which the included SoftcoreLennardJonesForce instance will belong.

    """
    def __init__(self, system, solute_atoms, forceGroup=0):
        super().__init__(system)
        nonbonded = self.getForce(atomsmm.findNonbondedForce(self))

        # Treat all solute-solute interactions as exceptions:
        existing_exceptions = []
        for index in range(nonbonded.getNumExceptions()):
            i, j, _, _, _ = nonbonded.getExceptionParameters(index)
            existing_exceptions.append(set([i, j]))
        for i, j in itertools.combinations(solute_atoms, 2):
            if set([i, j]) not in existing_exceptions:
                q1, sig1, eps1 = nonbonded.getParticleParameters(i)
                q2, sig2, eps2 = nonbonded.getParticleParameters(j)
                nonbonded.addException(i, j, q1*q2, (sig1 + sig2)/2, np.sqrt(eps1*eps2))

        # Include softcore Lennard-Jones interactions:
        rcut = nonbonded.getCutoffDistance()
        rswitch = nonbonded.getSwitchingDistance() if nonbonded.getUseSwitchingFunction() else None
        softcore = atomsmm.SoftcoreLennardJonesForce(rcut, rswitch, 'lambda_vdw')
        softcore.importFrom(nonbonded)
        solvent_atoms = set(range(nonbonded.getNumParticles())) - solute_atoms
        softcore.addInteractionGroup(solute_atoms, solvent_atoms)
        softcore.setForceGroup(forceGroup)
        softcore.addTo(self)

        # Turn off solute van der Waals interactions & scale solute charges w/ lambda_coul:
        nonbonded.addGlobalParameter('lambda_coul', 1.0)
        for i in solute_atoms:
            q, sig, eps = nonbonded.getParticleParameters(i)
            nonbonded.setParticleParameters(i, 0.0, 1.0, 0.0)
            if q/q.unit != 0.0:
                nonbonded.addParticleParameterOffset('lambda_coul', i, q, 0.0, 0.0)


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
                virial = atomsmm.forces._AtomsMM_CustomNonbondedForce(expression, rcut, rswitch, usesCharges=False)
                virial.importFrom(nonbonded)
                virial.setForceGroup(dispersionGroup)
                self.addForce(virial)
                exceptions = atomsmm.forces._AtomsMM_CustomBondForce(expression, usesCharges=False)
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
