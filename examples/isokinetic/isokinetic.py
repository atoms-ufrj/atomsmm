from __future__ import print_function

from simtk import openmm
from simtk import unit
from simtk.openmm import app
from sys import stdout

import atomsmm
from atomsmm import propagators as ppg
from openmmtools import integrators

# platform = 'Reference'
platform = 'OpenCL'
properties = dict(CUDA=dict(Precision = 'mixed'), Reference=dict(), OpenCL=dict())
nsteps = 100
ndisp = 10
seed = 5623
temp = 300*unit.kelvin
dt = 90*unit.femtoseconds
loops = [6, 30, 1]
rswitchIn = 5.0*unit.angstroms
rcutIn = 8.0*unit.angstroms
rswitch = 11.4*unit.angstroms
rcut = 12.4*unit.angstroms
tau = 10*unit.femtoseconds
gamma = 0.1/unit.femtoseconds
shift = False
forceSwitch = True

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
innerForce = atomsmm.NearNonbondedForce(rcutIn, rswitchIn, shift, forceSwitch).setForceGroup(1)
outerForce = atomsmm.FarNonbondedForce(innerForce, rcut, rswitch).setForceGroup(2)
for force in [exceptions, innerForce, outerForce]:
    force.importFrom(nbforce)
    force.addTo(system)

print("\nSplit model:")
for (term, energy) in atomsmm.utils.splitPotentialEnergy(system, pdb.topology, pdb.positions).items():
    print(term + ':', energy)

integrator = atomsmm.SIN_R_Integrator(dt, loops, temp, tau, gamma, location="center")
integrator.setRandomNumberSeed(seed)

print(integrator)

simulation = app.Simulation(pdb.topology, system, integrator,
                            openmm.Platform.getPlatformByName(platform), properties[platform])
simulation.context.setPositions(pdb.positions)
simulation.context.setVelocitiesToTemperature(300*unit.kelvin, seed)

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

print("\nSplit model:")
state = simulation.context.getState(getPositions=True)
for (term, energy) in atomsmm.utils.splitPotentialEnergy(simulation.context.getSystem(), pdb.topology, state.getPositions()).items():
    print(term + ':', energy)
