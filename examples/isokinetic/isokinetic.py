from __future__ import print_function

from simtk import openmm
from simtk import unit
from simtk.openmm import app
from sys import stdout

import atomsmm
from atomsmm import propagators as ppg
from openmmtools import integrators

# platform = 'Reference'
platform = 'CUDA'
properties = dict(CUDA=dict(Precision = 'mixed'), Reference=dict())
nsteps = 100
ndisp = 1
seed = 5623
temp = 300*unit.kelvin
dt = 0.5*unit.femtoseconds
loops = [1, 1, 1]
rswitchIn = 4.0*unit.angstroms
rcutIn = 6.0*unit.angstroms
rswitch = 8.0*unit.angstroms
rcut = 10*unit.angstroms
tau = 10*unit.femtoseconds
friction = 1/tau
shift = True

case = 'q-SPC-FW'
# case = 'emim_BCN4_Jiung2014'

pdb = app.PDBFile('../../tests/data/%s.pdb' % case)
forcefield = app.ForceField('../../tests/data/%s.xml' % case)
system = forcefield.createSystem(pdb.topology,
                                 nonbondedMethod=openmm.app.PME,
                                 nonbondedCutoff=rcut,
                                 rigidWater=False,
                                 constraints=None,
                                 removeCMMotion=False)

nbforceIndex = atomsmm.findNonbondedForce(system)
nbforce = system.getForce(nbforceIndex)
nbforce.setSwitchingDistance(rswitch)
nbforce.setUseSwitchingFunction(True)

print("Original model:")
for (term, energy) in atomsmm.utils.splitPotentialEnergy(system, pdb.topology, pdb.positions).items():
    print(term + ':', energy)

nbforce = atomsmm.hijackForce(system, nbforceIndex)
exceptions = atomsmm.NonbondedExceptionsForce().setForceGroup(0)
innerForce = atomsmm.NearNonbondedForce(rcutIn, rswitchIn, shift).setForceGroup(1)
outerForce = atomsmm.FarNonbondedForce(innerForce, rcut, rswitch).setForceGroup(2)
for force in [exceptions, innerForce, outerForce]:
    force.importFrom(nbforce)
    force.addTo(system)

print("\nSplit model:")
for (term, energy) in atomsmm.utils.splitPotentialEnergy(system, pdb.topology, pdb.positions).items():
    print(term + ':', energy)

dof = atomsmm.utils.countDegreesOfFreedom(system)
thermostat = ppg.VelocityRescalingPropagator(temp, dof, tau)

# integrator = openmm.LangevinIntegrator(temp, 1.0/unit.picoseconds, dt)
# dof = atomsmm.utils.countDegreesOfFreedom(system) - 3
# thermostat = ppg.NoseHooverPropagator(temp, dof, tau, nloops=1)
# thermostat = ppg.SuzukiYoshidaPropagator(thermostat)
# NVE = ppg.TrotterSuzukiPropagator(ppg.TranslationPropagator(constrained=False),
                            #    ppg.BoostPropagator(constrained=False))
# integrator = atomsmm.GlobalThermostatIntegrator(dt, NVE, thermostat)
integrator = ppg.RespaPropagator(loops, crust=thermostat).integrator(dt)

integrator = atomsmm.SIN_R_Integrator(dt, loops, temp, tau)
integrator.setRandomNumberSeed(seed)

print(integrator)
# print(integrator.getGlobalVariableByName("kT"))

simulation = app.Simulation(pdb.topology, system, integrator,
                            openmm.Platform.getPlatformByName(platform), properties[platform])
simulation.context.setPositions(pdb.positions)
integrator.initializeVelocities(simulation.context, 300*unit.kelvin)
# simulation.context.setVelocitiesToTemperature(300*unit.kelvin, seed)

# integrator.check(simulation.context)
# state = simulation.context.getState(getVelocities=True)
# masses = [system.getParticleMass(i) for i in range(system.getNumParticles())]
# for (m, v) in zip(masses, state.getVelocities()):
#     print((m*(v[0]**2+v[1]**2+v[2]**2)/(unit.BOLTZMANN_CONSTANT_kB*unit.AVOGADRO_CONSTANT_NA)).in_units_of(unit.kelvin))

outputs = [stdout, 'output.csv']
separators = ['\t', ',']
for (out, sep) in zip(outputs, separators):
    simulation.reporters.append(openmm.app.StateDataReporter(out, ndisp,
        step=True,
        potentialEnergy=True,
        kineticEnergy=True,
        totalEnergy=True,
        temperature=True,
        remainingTime=True,
        speed=True,
        totalSteps=nsteps,
        separator=sep))
# simulation.reporters.append(openmm.app.PDBReporter('output.pdb', nsteps))

# integrator.check(simulation.context)

print('Running Production...')
simulation.step(nsteps)

integrator.check(simulation.context)
# masses = [system.getParticleMass(i) for i in range(system.getNumParticles())]
# state = simulation.context.getState(getVelocities=True)
# v1s = simulation.integrator.getPerDofVariableByName('v1')
# Q1 = simulation.integrator.getGlobalVariableByName('Q1')
# for (m, v, v1) in zip(masses, state.getVelocities(), v1s):
#     for i in range(3):
#         print((m/unit.dalton)*(v[i]*unit.picoseconds/unit.nanometer)**2, 0.5*Q1*v1[i]*v1[i])
