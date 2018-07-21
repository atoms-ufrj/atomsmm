from __future__ import print_function

import pytest
from simtk import openmm
from simtk import unit
from simtk.openmm import app

import atomsmm


def execute(OuterForceType, shifted, target):
    rswitch_inner = 6.5*unit.angstroms
    rcut_inner = 7.0*unit.angstroms
    rswitch = 9.5*unit.angstroms
    rcut = 10*unit.angstroms
    case = 'tests/data/q-SPC-FW'
    pdb = app.PDBFile(case + '.pdb')
    forcefield = app.ForceField(case + '.xml')
    system = forcefield.createSystem(pdb.topology)
    reference = atomsmm.HijackNonbondedForce(system)
    innerforce = atomsmm.InnerRespaForce(rswitch_inner, rcut_inner, shifted).setForceGroup(1)
    innerforce.importFrom(reference).addTo(system)
    outerforce = OuterForceType(rswitch, rcut, innerforce).setForceGroup(2)
    outerforce.importFrom(reference).addTo(system)
    integrator = openmm.VerletIntegrator(0.0*unit.femtoseconds)
    platform = openmm.Platform.getPlatformByName('Reference')
    simulation = app.Simulation(pdb.topology, system, integrator, platform)
    simulation.context.setPositions(pdb.positions)
    for i in range(2):
        state = simulation.context.getState(getEnergy=True, groups=set([i+1]))
        potential = state.getPotentialEnergy()
        assert potential/potential.unit == pytest.approx(target[i])


def test_unshifted():
    execute(atomsmm.OuterRespaForce, False, [26864.363518197402, -24067.694751364364])


def test_shifted():
    execute(atomsmm.OuterRespaForce, True, [3934.08345914871, -7877.112930968626])
