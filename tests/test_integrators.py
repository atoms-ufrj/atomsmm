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


def all_subclasses(cls):
    subclasses = []
    for subclass in cls.__subclasses__():
        subclasses.append(subclass)
        subclasses.extend(all_subclasses(subclass))
    return subclasses


def test_variable_names():
    propagators = list()
    for cls in all_subclasses(atomsmm.propagators.Propagator):
        obj = cls.__new__(cls)
        obj.globalVariables = obj.perDofVariables = dict()
        obj.persistent = None
        obj.declareVariables()
        propagators.append(obj)
    for (i, a) in enumerate(propagators[:-1]):
        if a.persistent:
            for b in propagators[i+1:]:
                if b.persistent:
                    print(a.__class__.__name__, b.__class__.__name__)
                    assert set(a.persistent).isdisjoint(set(b.persistent))


def test_VelocityVerlet():
    NVE = atomsmm.VelocityVerletPropagator()
    integrator = atomsmm.GlobalThermostatIntegrator(1*unit.femtoseconds, NVE)
    execute(integrator, -5156.33314554173)


def test_Respa():
    NVE = atomsmm.RespaPropagator([1])
    integrator = atomsmm.GlobalThermostatIntegrator(1*unit.femtoseconds, NVE)
    execute(integrator, -5156.33314554173)


def test_BussiThermostat():
    system, positions, topology = readSystem('emim_BCN4_Jiung2014')
    dof = atomsmm.countDegreesOfFreedom(system)
    NVE = atomsmm.VelocityVerletPropagator()
    thermostat = atomsmm.BussiThermostatPropagator(300*unit.kelvin, 0.1*unit.picoseconds, dof)
    integrator = atomsmm.GlobalThermostatIntegrator(1*unit.femtoseconds, NVE, thermostat, 1)
    execute(integrator, -5155.97602603153)
