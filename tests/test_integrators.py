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
    nve = atomsmm.VelocityVerlet()
    integrator = atomsmm.GlobalThermostatIntegrator(1*unit.femtoseconds, nve)
    execute(integrator, -5156.33314554173)


def test_BussiDonadioParrinelloIntegrator():
    system, positions, topology = readSystem('emim_BCN4_Jiung2014')
    dof = atomsmm.countDegreesOfFreedom(system)
    nve = atomsmm.VelocityVerlet()
    thermostat = atomsmm.BussiDonadioParrinelloThermostat(300*unit.kelvin, 0.1*unit.picoseconds, dof)
    integrator = atomsmm.GlobalThermostatIntegrator(1*unit.femtoseconds, nve, thermostat, 1)
    execute(integrator, -5155.97602603153)
