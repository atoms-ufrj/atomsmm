from __future__ import print_function

import pytest
from simtk import openmm
from simtk import unit
from simtk.openmm import app

import atomsmm


def potentialEnergy(system, pdb):
    integrator = openmm.VerletIntegrator(0.0*unit.femtoseconds)
    platform = openmm.Platform.getPlatformByName('Reference')
    simulation = app.Simulation(pdb.topology, system, integrator, platform)
    simulation.context.setPositions(pdb.positions)
    state = simulation.context.getState(getEnergy=True)
    return state.getPotentialEnergy()


def execute(OuterForceType, shifted):
    rswitch_inner = 6.5*unit.angstroms
    rcut_inner = 7.0*unit.angstroms
    rswitch = 9.5*unit.angstroms
    rcut = 10*unit.angstroms
    case = 'tests/data/q-SPC-FW'
    pdb = app.PDBFile(case + '.pdb')
    forcefield = app.ForceField(case + '.xml')

    system = forcefield.createSystem(pdb.topology)
    nbforce = atomsmm.hijackNonbondedForce(system)
    innerforce = atomsmm.InnerRespaForce(rcut_inner, rswitch_inner, shifted).setForceGroup(1)
    innerforce.importFrom(nbforce).addTo(system)
    outerforce = OuterForceType(innerforce, rcut, rswitch).setForceGroup(2)
    outerforce.importFrom(nbforce).addTo(system)

    refsys = forcefield.createSystem(pdb.topology,
                                     nonbondedMethod=openmm.app.PME,
                                     nonbondedCutoff=rcut,
                                     removeCMMotion=True)
    force = refsys.getForce(refsys.getNumForces()-2)
    force.setUseSwitchingFunction(True)
    force.setSwitchingDistance(rswitch)

    potential = potentialEnergy(system, pdb)
    refpot = potentialEnergy(refsys, pdb)
    assert potential/potential.unit == pytest.approx(refpot/refpot.unit)


def test_unshifted():
    execute(atomsmm.OuterRespaForce, False)


def test_shifted():
    execute(atomsmm.OuterRespaForce, True)
