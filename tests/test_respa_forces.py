from __future__ import print_function

import pytest
from simtk import openmm
from simtk import unit
from simtk.openmm import app

import atomsmm


def executeNearForceTest(adjustment, target):
    rcut = 10*unit.angstroms
    rswitch = 9.5*unit.angstroms
    case = 'tests/data/q-SPC-FW'
    pdb = app.PDBFile(case + '.pdb')
    forcefield = app.ForceField(case + '.xml')
    system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.CutoffPeriodic)
    force = atomsmm.NearNonbondedForce(rcut, rswitch, adjustment)
    force.importFrom(atomsmm.hijackForce(system, atomsmm.findNonbondedForce(system))).addTo(system)
    integrator = openmm.VerletIntegrator(0.0*unit.femtoseconds)
    platform = openmm.Platform.getPlatformByName('Reference')
    simulation = app.Simulation(pdb.topology, system, integrator, platform)
    simulation.context.setPositions(pdb.positions)
    state = simulation.context.getState(getEnergy=True)
    potential = state.getPotentialEnergy()
    assert potential/potential.unit == pytest.approx(target)


def test_unshifted_near():
    executeNearForceTest(None, -24955.845391462222)


def test_shifted_near():
    executeNearForceTest("shift", -26451.885982885935)


def test_force_swiched_near():
    executeNearForceTest("force-switch", -25265.454307792825)


def executeFarForceTest(OuterForceType, adjustment):
    rswitch_inner = 6.5*unit.angstroms
    rcut_inner = 7.0*unit.angstroms
    rswitch = 9.5*unit.angstroms
    rcut = 10*unit.angstroms
    case = 'tests/data/q-SPC-FW'
    pdb = app.PDBFile(case + '.pdb')
    forcefield = app.ForceField(case + '.xml')

    system = forcefield.createSystem(pdb.topology)
    nbforce = atomsmm.hijackForce(system, atomsmm.findNonbondedForce(system))
    innerforce = atomsmm.NearNonbondedForce(rcut_inner, rswitch_inner, adjustment).setForceGroup(1)
    innerforce.importFrom(nbforce).addTo(system)
    outerforce = OuterForceType(innerforce, rcut, rswitch).setForceGroup(2)
    outerforce.importFrom(nbforce).addTo(system)
    potential = atomsmm.splitPotentialEnergy(system, pdb.topology, pdb.positions)["Total"]

    refsys = forcefield.createSystem(pdb.topology,
                                     nonbondedMethod=openmm.app.PME,
                                     nonbondedCutoff=rcut,
                                     removeCMMotion=True)
    force = refsys.getForce(refsys.getNumForces()-2)
    force.setUseSwitchingFunction(True)
    force.setSwitchingDistance(rswitch)
    force.setEwaldErrorTolerance(1E-5)
    refpot = atomsmm.splitPotentialEnergy(refsys, pdb.topology, pdb.positions)["Total"]

    assert potential/potential.unit == pytest.approx(refpot/refpot.unit)


def test_unshifted_far():
    executeFarForceTest(atomsmm.FarNonbondedForce, None)


def test_shifted_far():
    executeFarForceTest(atomsmm.FarNonbondedForce, "shift")


def test_force_swiched_far():
    executeFarForceTest(atomsmm.FarNonbondedForce, "force-switch")
