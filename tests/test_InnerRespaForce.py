from __future__ import print_function

import pytest
from simtk import openmm
from simtk import unit
from simtk.openmm import app

import atomsmm


def execute(shifted, target):
    rcut = 10*unit.angstroms
    rswitch = 9.5*unit.angstroms
    case = 'tests/data/q-SPC-FW'
    pdb = app.PDBFile(case + '.pdb')
    forcefield = app.ForceField(case + '.xml')
    system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.CutoffPeriodic)
    force = atomsmm.InnerRespaForce(rcut, rswitch, shifted)
    force.importFrom(atomsmm.hijackForce(system, atomsmm.findNonbondedForce(system))).addTo(system)
    integrator = openmm.VerletIntegrator(0.0*unit.femtoseconds)
    platform = openmm.Platform.getPlatformByName('Reference')
    simulation = app.Simulation(pdb.topology, system, integrator, platform)
    simulation.context.setPositions(pdb.positions)
    state = simulation.context.getState(getEnergy=True)
    potential = state.getPotentialEnergy()
    assert potential/potential.unit == pytest.approx(target)


def test_unshifted():
    execute(False, 11517.016971940495)


def test_shifted():
    execute(True, 3182.9377183815345)
