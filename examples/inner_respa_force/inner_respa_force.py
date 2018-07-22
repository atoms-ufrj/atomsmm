from __future__ import print_function

from simtk import openmm
from simtk import unit
from simtk.openmm import app
from sys import stdout

import atomsmm

nsteps = 2000
ndisp = 10
dt = 1.0*unit.femtoseconds
rcut = 10*unit.angstroms
rswitch = rcut - 1.0*unit.angstroms
shift = True

# case = 'q-SPC-FW'
case = 'emim_BCN4_Jiung2014'

pdb = app.PDBFile('../../tests/data/%s.pdb' % case)
forcefield = app.ForceField('../../tests/data/%s.xml' % case)
system = forcefield.createSystem(pdb.topology, rigid_water=True)
nbforce = atomsmm.HijackNonbondedForce(system)
atomsmm.InnerRespaForce(rswitch, rcut, shift).importFrom(nbforce).addTo(system)
integrator = openmm.VerletIntegrator(dt)
platform = openmm.Platform.getPlatformByName('CUDA')
simulation = app.Simulation(pdb.topology, system, integrator, platform)
simulation.context.setPositions(pdb.positions)

outputs = [stdout, 'output.csv']
# outputs = [stdout]
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
