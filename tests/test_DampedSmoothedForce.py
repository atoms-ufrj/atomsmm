from __future__ import print_function

from simtk import openmm
from simtk import unit
from simtk.openmm import app

import atomsmm


def test_main():
    rcut = 10*unit.angstroms
    rswitch = 9.5*unit.angstroms
    alpha = 0.29/unit.angstroms

    pdb = app.PDBFile('tests/data/q-SPC-FW.pdb')
    forcefield = app.ForceField('tests/data/q-SPC-FW.xml')
    system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.CutoffPeriodic)
    integrator = openmm.VerletIntegrator(0.0*unit.femtoseconds)
    platform = openmm.Platform.getPlatformByName('Reference')
    atomsmm.DampedSmoothedForce(alpha, rswitch, rcut, degree=2).addTo(system, replace=True)
    simulation = app.Simulation(pdb.topology, system, integrator, platform)
    simulation.context.setPositions(pdb.positions)
    state = simulation.context.getState(getEnergy=True)
    potential = state.getPotentialEnergy()
    assert potential == 1765.940220790936*unit.kilojoules_per_mole
