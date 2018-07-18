from __future__ import print_function

import pytest
from simtk import openmm
from simtk import unit
from simtk.openmm import app

import atomsmm


def execute(subtract, shift, target):
    rcut = 10*unit.angstroms
    rswitch = 9.5*unit.angstroms
    pdb = app.PDBFile('tests/data/q-SPC-FW.pdb')
    forcefield = app.ForceField('tests/data/q-SPC-FW.xml')
    system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.CutoffPeriodic)
    force = atomsmm.InnerRespaForce(rswitch, rcut, subtract, shift)
    force.addTo(system, replace=True)
    integrator = openmm.VerletIntegrator(0.0*unit.femtoseconds)
    platform = openmm.Platform.getPlatformByName('Reference')
    simulation = app.Simulation(pdb.topology, system, integrator, platform)
    simulation.context.setPositions(pdb.positions)
    state = simulation.context.getState(getEnergy=True)
    potential = state.getPotentialEnergy()
    assert potential/potential.unit == pytest.approx(target)


def test_defaults():
    execute(False, False, 11517.016971940495)


def test_subtract():
    execute(True, False, -11517.016971940495)


def test_shift():
    execute(False, True, 3097.9538355702375)
