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


def execute(integrator, target):
    print(integrator)
    system, positions, topology = readSystem('emim_BCN4_Jiung2014')
    nbforce = atomsmm.findNonbondedForce(system)
    system.getForce(nbforce).setReciprocalSpaceForceGroup(1)
    platform = openmm.Platform.getPlatformByName('Reference')
    simulation = app.Simulation(topology, system, integrator, platform)
    simulation.context.setPositions(positions)
    simulation.context.setVelocitiesToTemperature(300*unit.kelvin, 1)
    simulation.step(5)
    state = simulation.context.getState(getEnergy=True)
    potential = state.getPotentialEnergy()
    print(potential, target)
    assert potential/potential.unit == pytest.approx(target)


def test_VelocityVerlet():
    NVE = atomsmm.VelocityVerletPropagator()
    integrator = atomsmm.GlobalThermostatIntegrator(1*unit.femtoseconds, NVE)
    execute(integrator, -13069.385589639898)


def test_Respa():
    boost = atomsmm.propagators.VelocityBoostPropagator(constrained=True)
    move = atomsmm.propagators.TranslationPropagator(constrained=True)
    NVE = atomsmm.RespaPropagator([4, 1], boost=boost, move=move)
    integrator = atomsmm.GlobalThermostatIntegrator(1*unit.femtoseconds, NVE)
    execute(integrator, -13127.4164665226)


def test_BussiThermostat():
    system, positions, topology = readSystem('emim_BCN4_Jiung2014')
    dof = atomsmm.countDegreesOfFreedom(system)
    NVE = atomsmm.VelocityVerletPropagator()
    thermostat = atomsmm.VelocityRescalingPropagator(300*unit.kelvin, dof, 0.1*unit.picoseconds)
    integrator = atomsmm.GlobalThermostatIntegrator(1*unit.femtoseconds, NVE, thermostat)
    integrator.setRandomNumberSeed(1)
    execute(integrator, -13063.298646767591)


def test_Chained():
    system, positions, topology = readSystem('emim_BCN4_Jiung2014')
    dof = atomsmm.countDegreesOfFreedom(system)
    NVE = atomsmm.VelocityVerletPropagator()
    thermostat = atomsmm.VelocityRescalingPropagator(300*unit.kelvin, dof, 0.1*unit.picoseconds)
    integrator = atomsmm.ChainedPropagator(NVE, thermostat).integrator(1*unit.femtoseconds)
    integrator.setRandomNumberSeed(1)
    execute(integrator, -13063.68000796426)


def test_TrotterSuzuki():
    system, positions, topology = readSystem('emim_BCN4_Jiung2014')
    dof = atomsmm.countDegreesOfFreedom(system)
    NVE = atomsmm.VelocityVerletPropagator()
    thermostat = atomsmm.VelocityRescalingPropagator(300*unit.kelvin, dof, 0.1*unit.picoseconds)
    combined = atomsmm.TrotterSuzukiPropagator(NVE, thermostat)
    integrator = combined.integrator(1*unit.femtoseconds)
    integrator.setRandomNumberSeed(1)
    execute(integrator, -13063.298646767591)


def test_NoseHooverPropagator():
    system, positions, topology = readSystem('emim_BCN4_Jiung2014')
    dof = atomsmm.countDegreesOfFreedom(system)
    NVE = atomsmm.VelocityVerletPropagator()
    thermostat = atomsmm.NoseHooverPropagator(300*unit.kelvin, dof, 10*unit.femtoseconds, 2)
    thermostat = atomsmm.SuzukiYoshidaPropagator(thermostat, 3)
    combined = atomsmm.TrotterSuzukiPropagator(NVE, thermostat)
    integrator = combined.integrator(1*unit.femtoseconds)
    integrator.setRandomNumberSeed(1)
    execute(integrator, -13067.685398497035)


def test_MassiveNoseHooverPropagator():
    system, positions, topology = readSystem('emim_BCN4_Jiung2014')
    NVE = atomsmm.VelocityVerletPropagator()
    thermostat = atomsmm.propagators.MassiveNoseHooverPropagator(300*unit.kelvin, 10*unit.femtoseconds, 1)
    combined = atomsmm.TrotterSuzukiPropagator(NVE, thermostat)
    integrator = combined.integrator(1*unit.femtoseconds)
    integrator.setRandomNumberSeed(1)
    execute(integrator, -13484.829386266081)
