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
                                     nonbondedMethod=app.PME,
                                     constraints=None,
                                     rigidWater=False,
                                     removeCMMotion=False)
    return system, pdb.positions, pdb.topology


def test_pressure_with_bath_temperature():
    system, positions, topology = readSystem('q-SPC-FW')
    platform = openmm.Platform.getPlatformByName('Reference')
    computer = atomsmm.PressureComputer(system, topology, platform, temperature=300*unit.kelvin)
    context = openmm.Context(system, openmm.CustomIntegrator(0))
    context.setPositions(positions)
    state = context.getState(getPositions=True, getVelocities=True, getForces=True)
    computer.import_configuration(state)
    atomic_virial = computer.get_atomic_virial()
    assert atomic_virial/atomic_virial.unit == pytest.approx(-11661.776513617617)
    atomic_pressure = computer.get_atomic_pressure()
    assert atomic_pressure/atomic_pressure.unit == pytest.approx(-58.68294207631704)
    molecular_virial = computer.get_molecular_virial(state.getForces())
    assert molecular_virial/molecular_virial.unit == pytest.approx(-5418.7207680747715)
    molecular_pressure = computer.get_molecular_pressure(state.getForces())
    assert molecular_pressure/molecular_pressure.unit == pytest.approx(-554.9843622696719)


def test_pressure_with_kinetic_temperature():
    system, positions, topology = readSystem('q-SPC-FW')
    platform = openmm.Platform.getPlatformByName('Reference')
    computer = atomsmm.PressureComputer(system, topology, platform)
    context = openmm.Context(system, openmm.CustomIntegrator(0))
    context.setPositions(positions)
    context.setVelocitiesToTemperature(300*unit.kelvin, 1234)
    state = context.getState(getPositions=True, getVelocities=True, getForces=True)
    computer.import_configuration(state)
    atomic_virial = computer.get_atomic_virial()
    assert atomic_virial/atomic_virial.unit == pytest.approx(-11661.776513617617)
    atomic_pressure = computer.get_atomic_pressure()
    assert atomic_pressure/atomic_pressure.unit == pytest.approx(-86.99379079995832)
    molecular_virial = computer.get_molecular_virial(state.getForces())
    assert molecular_virial/molecular_virial.unit == pytest.approx(-5418.7207680747715)
    molecular_pressure = computer.get_molecular_pressure(state.getForces())
    assert molecular_pressure/molecular_pressure.unit == pytest.approx(-539.0399787048871)


def test_pressure_with_exceptions():
    system, positions, topology = readSystem('emim_BCN4_Jiung2014')
    platform = openmm.Platform.getPlatformByName('Reference')
    computer = atomsmm.PressureComputer(system, topology, platform, temperature=300*unit.kelvin)
    context = openmm.Context(system, openmm.CustomIntegrator(0))
    context.setPositions(positions)
    state = context.getState(getPositions=True, getVelocities=True, getForces=True)
    computer.import_configuration(state)
    atomic_virial = computer.get_atomic_virial()
    assert atomic_virial/atomic_virial.unit == pytest.approx(-22827.520859196087)
    atomic_pressure = computer.get_atomic_pressure()
    assert atomic_pressure/atomic_pressure.unit == pytest.approx(-282.7308090971683)
    molecular_virial = computer.get_molecular_virial(state.getForces())
    assert molecular_virial/molecular_virial.unit == pytest.approx(-23273.15088720543)
    molecular_pressure = computer.get_molecular_pressure(state.getForces())
    assert molecular_pressure/molecular_pressure.unit == pytest.approx(-3283.592285100925)
