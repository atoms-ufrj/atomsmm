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
    nvt_integrator = atomsmm.SIN_R_Integrator(
        1*unit.femtoseconds,
        [1],
        300*unit.kelvin,
        10*unit.femtoseconds,
        0.1/unit.femtoseconds,
    )
    system, positions, topology = readSystem('hydroxyethylaminoanthraquinone-in-water')
    residues = [atom.residue.name for atom in topology.atoms()]
    solute = set(i for (i, name) in enumerate(residues) if name == 'aaa')
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
    assert potential/potential.unit == pytest.approx(-61952.0479169265)
