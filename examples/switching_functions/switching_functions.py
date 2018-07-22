from __future__ import print_function

from simtk import openmm
from simtk import unit
from simtk.openmm import app
from sys import stdout

import atomsmm

nsteps = 5000
ndisp = 10
temp = 30*unit.kelvin
dt = 1.0*unit.femtoseconds
rcut = 10*unit.angstroms
rswitch = 9.5*unit.angstroms
alpha = 0.29/unit.angstroms
degree = 2

# case = 'q-SPC-FW'
case = 'emim_BCN4_Jiung2014'

pdb = app.PDBFile('../../tests/data/%s.pdb' % case)
forcefield = app.ForceField('../../tests/data/%s.xml' % case)
system = forcefield.createSystem(pdb.topology, rigid_water=True)
nbforce = atomsmm.utils.HijackNonbondedForce(system)
atomsmm.DampedSmoothedForce(alpha, rswitch, rcut, degree).importFrom(nbforce).addTo(system)
integrator = openmm.LangevinIntegrator(temp, 1.0/unit.picosecond, dt)
platform = openmm.Platform.getPlatformByName('CUDA')
simulation = app.Simulation(pdb.topology, system, integrator, platform)
simulation.context.setPositions(pdb.positions)

# outputs = [stdout, 'degree_%d.csv' % degree]
outputs = [stdout]
separators = ['\t', ',']
for (out, sep) in zip(outputs, separators):
    simulation.reporters.append(openmm.app.StateDataReporter(out, ndisp,
        step=True,
        temperature=True,
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
