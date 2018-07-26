from __future__ import print_function

import pytest
from simtk import openmm
from simtk import unit
from simtk.openmm import app

import atomsmm


def execute(integrator, target):
    rcut = 10*unit.angstroms
    rswitch = 9.5*unit.angstroms
    alpha = 0.29/unit.angstroms
    case = 'tests/data/emim_BCN4_Jiung2014'
    pdb = app.PDBFile(case + '.pdb')
    forcefield = app.ForceField(case + '.xml')
    system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.CutoffPeriodic,
                                     constraints=app.HBonds)
    force = atomsmm.DampedSmoothedForce(alpha, rcut, rswitch, degree=2)
    force.importFrom(atomsmm.hijackForce(system, atomsmm.findNonbondedForce(system))).addTo(system)
    platform = openmm.Platform.getPlatformByName('Reference')
    simulation = app.Simulation(pdb.topology, system, integrator, platform)
    simulation.context.setPositions(pdb.positions)
    simulation.step(2)
    state = simulation.context.getState(getEnergy=True)
    potential = state.getPotentialEnergy()
    print(potential)
    assert potential/potential.unit == pytest.approx(target)


def test_VelocityVerletIntegrator():
    integrator = atomsmm.VelocityVerletIntegrator(1.0*unit.femtoseconds)
    execute(integrator, -6139.391902890777)
