from __future__ import print_function

from simtk import openmm
from simtk import unit
from simtk.openmm import app
from sys import stdout

import atomsmm

nsteps = 2000
ndisp = 10
dt = 1.0*unit.femtoseconds
rswitchIn = 6.0*unit.angstroms
rcutIn = 7.0*unit.angstroms
rswitch = 9.0*unit.angstroms
rcut = 10*unit.angstroms
shift = True
mts = False
# mts = True

case = 'q-SPC-FW'
# case = 'emim_BCN4_Jiung2014'

pdb = app.PDBFile('../../tests/data/%s.pdb' % case)
forcefield = app.ForceField('../../tests/data/%s.xml' % case)
system = forcefield.createSystem(pdb.topology, nonbondedMethod=openmm.app.PME,
                                 nonbondedCutoff=rcut, rigidWater=False)

if mts:
    nbforce = atomsmm.hijackNonbondedForce(system)
    exceptions = atomsmm.Force().includeExceptions().setForceGroup(0)
    innerForce = atomsmm.InnerRespaForce(rcutIn, rswitchIn, shift).setForceGroup(1)
    outerForce = atomsmm.OuterRespaForce(innerForce, rcut, rswitch).setForceGroup(2)
    for force in [exceptions, innerForce, outerForce]:
        force.importFrom(nbforce)
        force.addTo(system)
    integrator = openmm.VerletIntegrator(dt)  # CHANGE HERE
else:
    integrator = openmm.VerletIntegrator(dt)

platform = openmm.Platform.getPlatformByName('CUDA')
simulation = app.Simulation(pdb.topology, system, integrator, platform)
simulation.context.setPositions(pdb.positions)

outputs = [stdout, 'output.csv']
separators = ['\t', ',']
for (out, sep) in zip(outputs, separators):
    simulation.reporters.append(openmm.app.StateDataReporter(out, ndisp,
        step=True,
        potentialEnergy=True,
        kineticEnergy=True,
        totalEnergy=True,
        remainingTime=True,
        speed=True,
        totalSteps=nsteps,
        separator=sep))
# simulation.reporters.append(openmm.app.PDBReporter('output.pdb', nsteps))

print('Running Production...')
simulation.step(nsteps)
