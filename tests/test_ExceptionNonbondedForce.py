from __future__ import print_function

import pytest
from simtk import openmm
from simtk import unit
from simtk.openmm import app

import atomsmm


def test_exceptions():
    case = 'tests/data/emim_BCN4_Jiung2014'
    pdb = app.PDBFile(case + '.pdb')
    forcefield = app.ForceField(case + '.xml')
    system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.CutoffPeriodic)
    force = atomsmm.forces.Force().includeExceptions()
    force.importFrom(atomsmm.HijackNonbondedForce(system)).addTo(system)
    integrator = openmm.VerletIntegrator(0.0*unit.femtoseconds)
    platform = openmm.Platform.getPlatformByName('Reference')
    simulation = app.Simulation(pdb.topology, system, integrator, platform)
    simulation.context.setPositions(pdb.positions)
    state = simulation.context.getState(getEnergy=True)
    potential = state.getPotentialEnergy()
    assert potential/potential.unit == pytest.approx(-27616.298459208883)
