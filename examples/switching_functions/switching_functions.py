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
rswitch = 9.5*unit.angstroms
alpha = 0.29/unit.angstroms
degree = 1

pdb = app.PDBFile('../../tests/data/q-SPC-FW.pdb')
forcefield = app.ForceField('../../tests/data/q-SPC-FW.xml')
system = forcefield.createSystem(pdb.topology, rigid_water=True)
atomsmm.DampedSmoothedForce(alpha, rswitch, rcut, degree=degree).addTo(system, replace=True)
integrator = openmm.VerletIntegrator(dt)
platform = openmm.Platform.getPlatformByName('CUDA')
simulation = app.Simulation(pdb.topology, system, integrator, platform)
simulation.context.setPositions(pdb.positions)

# outputs = [stdout, 'degree_%d.csv' % degree]
outputs = [stdout]
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

print('Running Production...')
simulation.step(nsteps)
