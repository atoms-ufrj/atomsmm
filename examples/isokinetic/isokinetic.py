from __future__ import print_function

from simtk import openmm
from simtk import unit
from simtk.openmm import app
from sys import stdout

import atomsmm

nsteps = 1
ndisp = 100
seed = 5623
temp = 300*unit.kelvin
dt = 2.0*unit.femtoseconds
friction = 10/unit.picoseconds
rswitchIn = 6.0*unit.angstroms
rcutIn = 7.0*unit.angstroms
rswitch = 9.0*unit.angstroms
rcut = 10*unit.angstroms
shift = True

# case = 'q-SPC-FW'
case = 'emim_BCN4_Jiung2014'

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
nbforce.setUseSwitchingFunction(True)
nbforce.setSwitchingDistance(rswitch)
positions = atomsmm.TranslationPropagator()
integrator = atomsmm.SIN_R_Integrator(dt, temp, system.getNumParticles())

integrator.pretty_print()

for (term, energy) in atomsmm.utils.splitPotentialEnergy(system, pdb.topology, pdb.positions).items():
    print(term + ":", energy)

platform = openmm.Platform.getPlatformByName('CUDA')
properties = {"Precision": "mixed"}
simulation = app.Simulation(pdb.topology, system, integrator, platform, properties)
simulation.context.setPositions(pdb.positions)
integrator.initializeVelocities(simulation.context, 300*unit.kelvin)

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

print('Running Production...')
simulation.step(nsteps)
state = simulation.context.getState(getVelocities=True)
for v in state.getVelocities():
    print(v)
