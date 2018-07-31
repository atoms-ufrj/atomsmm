from __future__ import print_function

import pytest
from simtk import openmm
from simtk import unit
from simtk.openmm import app

import atomsmm


def readSystem(case):
    pdb = app.PDBFile('tests/data/%s.pdb' % case)
    forcefield = app.ForceField('tests/data/%s.xml' % case)
    system = forcefield.createSystem(pdb.topology,
                                     nonbondedMethod=app.CutoffPeriodic,
                                     constraints=app.HBonds)
    return system, pdb.positions, pdb.topology


def execute(integrator, target):
    rcut = 10*unit.angstroms
    rswitch = 9.5*unit.angstroms
    alpha = 0.29/unit.angstroms
    system, positions, topology = readSystem('emim_BCN4_Jiung2014')
    force = atomsmm.DampedSmoothedForce(alpha, rcut, rswitch, degree=2)
    force.importFrom(atomsmm.hijackForce(system, atomsmm.findNonbondedForce(system))).addTo(system)
    platform = openmm.Platform.getPlatformByName('Reference')
    simulation = app.Simulation(topology, system, integrator, platform)
    simulation.context.setPositions(positions)
    simulation.context.setVelocitiesToTemperature(300*unit.kelvin, 1)
    simulation.step(2)
    state = simulation.context.getState(getEnergy=True)
    potential = state.getPotentialEnergy()
    assert potential/potential.unit == pytest.approx(target)


def test_VelocityVerletIntegrator():
    integrator = atomsmm.VelocityVerletIntegrator(1.0*unit.femtoseconds)
    execute(integrator, -5156.33314554173)


def test_BussiDonadioParrinelloIntegrator():
    system, positions, topology = readSystem('emim_BCN4_Jiung2014')
    dof = atomsmm.countDegreesOfFreedom(system)
    integrator = atomsmm.BussiDonadioParrinelloIntegrator(300*unit.kelvin,
                                                          10/unit.picoseconds,
                                                          1*unit.femtoseconds,
                                                          dof, 1)
    execute(integrator, -5155.97602603153)
