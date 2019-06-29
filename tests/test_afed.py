from __future__ import print_function

import pytest
from simtk import openmm
from simtk import unit
from simtk.openmm import app

import atomsmm


def readSystem(case, constraints=app.HBonds):
    pdb = app.PDBFile('tests/data/%s.pdb' % case)
    forcefield = app.ForceField('tests/data/%s.xml' % case)
    system = forcefield.createSystem(pdb.topology,
                                     nonbondedMethod=app.PME,
                                     constraints=constraints,
                                     removeCMMotion=constraints is not None)
    return system, pdb.positions, pdb.topology


def test_AdiabaticFreeEnergyDynamicsIntegrator():
    system, positions, topology = readSystem('methane-in-water')
    nvt_integrator = atomsmm.propagators.TrotterSuzukiPropagator(
        atomsmm.propagators.VelocityVerletPropagator(),
        atomsmm.propagators.NoseHooverPropagator(
            300*unit.kelvin,
            atomsmm.countDegreesOfFreedom(system),
            10*unit.femtoseconds,
        ),
    ).integrator(1*unit.femtosecond)
    residues = [atom.residue.name for atom in topology.atoms()]
    solute = set(i for (i, name) in enumerate(residues) if name == 'C1')
    solvation_system = atomsmm.AlchemicalSystem(system, solute)
    integrator = atomsmm.integrators.AdiabaticFreeEnergyDynamicsIntegrator(
        nvt_integrator, 2, 'lambda_vdw', 1000, 5,
    )
    integrator.setRandomNumberSeed(1234)
    platform = openmm.Platform.getPlatformByName('Reference')
    simulation = app.Simulation(topology, solvation_system, integrator, platform)
    simulation.context.setPositions(positions)
    simulation.context.setVelocitiesToTemperature(300*unit.kelvin, 1234)
    simulation.step(5)
    state = simulation.context.getState(getEnergy=True)
    potential = state.getPotentialEnergy()
    assert potential/potential.unit == pytest.approx(-23132.97706420563)
