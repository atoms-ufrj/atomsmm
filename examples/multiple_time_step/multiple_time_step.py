from __future__ import print_function

from simtk import openmm
from simtk import unit
from simtk.openmm import app
from sys import stdout

import atomsmm

nsteps = 100
ndisp = 10
seed = 5623
temp = 300*unit.kelvin
dt = 1.0*unit.femtoseconds
tau = 10*unit.femtoseconds
rswitchIn = 6.0*unit.angstroms
rcutIn = 7.0*unit.angstroms
rswitch = 9.0*unit.angstroms
rcut = 10*unit.angstroms
shift = False
mts = False
mts = True

platform = 'CUDA'
properties = dict(CUDA=dict(Precision = 'mixed'), CPU=dict(), Reference=dict())

case = 'q-SPC-FW'
# case = 'emim_BCN4_Jiung2014'

pdb = app.PDBFile('../../tests/data/%s.pdb' % case)
forcefield = app.ForceField('../../tests/data/%s.xml' % case)
system = forcefield.createSystem(pdb.topology, nonbondedMethod=openmm.app.PME,
                                 nonbondedCutoff=rcut, rigidWater=False, constraints=None)

nbforceIndex = atomsmm.findNonbondedForce(system)
dof = atomsmm.countDegreesOfFreedom(system)
thermostat = atomsmm.NoseHooverPropagator(temp, dof, tau)
if mts:
    nbforce = atomsmm.hijackForce(system, nbforceIndex)
    exceptions = atomsmm.NonbondedExceptionsForce().setForceGroup(0)
    innerForce = atomsmm.NearNonbondedForce(rcutIn, rswitchIn, shift).setForceGroup(1)
    outerForce = atomsmm.FarNonbondedForce(innerForce, rcut, rswitch).setForceGroup(2)
    for force in [exceptions, innerForce, outerForce]:
        force.importFrom(nbforce)
        force.addTo(system)
    integrator = atomsmm.RespaPropagator([5, 2, 1], mantle=thermostat).integrator(dt)
else:
    nbforce = system.getForce(nbforceIndex)
    nbforce.setUseSwitchingFunction(True)
    nbforce.setSwitchingDistance(rswitch)
    NVE = atomsmm.VelocityVerletPropagator()
    integrator = atomsmm.GlobalThermostatIntegrator(dt, NVE, thermostat)

print(integrator)

for (term, energy) in atomsmm.utils.splitPotentialEnergy(system, pdb.topology, pdb.positions).items():
    print(term + ":", energy)

simulation = app.Simulation(pdb.topology, system, integrator,
                            openmm.Platform.getPlatformByName(platform), properties[platform])
simulation.context.setPositions(pdb.positions)
simulation.context.setVelocitiesToTemperature(300*unit.kelvin, seed)

state = simulation.context.getState(getEnergy=True)
print(state.getPotentialEnergy())

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

print('Running Production...')
simulation.step(nsteps)
