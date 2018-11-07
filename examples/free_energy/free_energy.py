from __future__ import print_function

from simtk import openmm
from simtk import unit
from simtk.openmm import app
from sys import stdout

import atomsmm
import numpy as np

platform = 'OpenCL'
properties = dict(CUDA=dict(Precision='mixed'), Reference=dict(), OpenCL=dict())

nprod = 30000
nequil = 0
ndisp = 10
seed = 5623

temp = 300*unit.kelvin
dt = 30*unit.femtoseconds
loops = [6, 10, 1]

rswitchIn = 4.0*unit.angstroms
rcutIn = 7.0*unit.angstroms
rswitch = 9.0*unit.angstroms
rcut = 10*unit.angstroms
tau = 10*unit.femtoseconds
gamma = 0.1/unit.femtoseconds
shift = True

case = 'water_benzene'
pdb = app.PDBFile('{}.pdb'.format(case))
forcefield = app.ForceField('{}.xml'.format(case))
system = forcefield.createSystem(pdb.topology,
                                 nonbondedMethod=openmm.app.PME,
                                 nonbondedCutoff=rcut,
                                 rigidWater=False,
                                 constraints=None,
                                 removeCMMotion=False)

nbforce = atomsmm.hijackForce(system, atomsmm.findNonbondedForce(system))
exceptions = atomsmm.NonbondedExceptionsForce().importFrom(nbforce)

softcore = atomsmm.SoftcoreLennardJonesForce(rcut, rswitch).importFrom(nbforce)
alchemical_atoms = set(range(6))
for index in alchemical_atoms:
    charge, sigma, epsilon = nbforce.getParticleParameters(index)
    nbforce.setParticleParameters(index, charge*0, sigma, epsilon*0)
all_atoms = set(range(system.getNumParticles()))
softcore.addInteractionGroup(alchemical_atoms, all_atoms - alchemical_atoms)

innerForce = atomsmm.NearNonbondedForce(rcutIn, rswitchIn, shift).importFrom(nbforce)
outerForce = atomsmm.FarNonbondedForce(innerForce, rcut, rswitch).importFrom(nbforce)

exceptions.setForceGroup(0).addTo(system)
innerForce.setForceGroup(1).addTo(system)
outerForce.setForceGroup(2).addTo(system)
softcore.setForceGroup(1).addTo(system)

# print("Initial energies:")
# for (term, energy) in atomsmm.utils.splitPotentialEnergy(system, pdb.topology, pdb.positions).items():
#     print(term + ':', energy)

integrator = atomsmm.SIN_R_Integrator(dt, loops, temp, tau, gamma)
integrator.setRandomNumberSeed(seed)

simulation = app.Simulation(pdb.topology, system, integrator,
                            openmm.Platform.getPlatformByName(platform), properties[platform])
simulation.context.setPositions(pdb.positions)
simulation.context.setVelocitiesToTemperature(temp, seed)

lambdas = [0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
free_energies = np.array([0.0000, 0.1151, 0.4008, 1.4484, 2.4603, 2.8056, 2.7540, 2.5040, 2.3294, 2.1389, 1.9325, 1.7103, 1.4762, 1.2302, 0.9762, 0.7143])

states = {"lambda": lambdas}
kT = (unit.BOLTZMANN_CONSTANT_kB*unit.AVOGADRO_CONSTANT_NA*temp).value_in_unit(unit.kilocalories_per_mole)
weights = free_energies/kT

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
        totalSteps=len(states)*(nequil+nprod),
        separator=sep))
# simulation.reporters.append(openmm.app.PDBReporter('output.pdb', nprod))

# for (i, value) in enumerate(states["lambda"]):
#     simulation.context.setParameter("lambda", value)
#     print("\nRunning equilibration for lambda={}:".format(value))
#     simulation.step(nequil)
#     simulation.reporters.append(atomsmm.MultistateEnergyReporter('state{}.csv'.format(i), ndisp, states))
#     print("\nRunning production for lambda={}:".format(value))
#     simulation.step(nprod)
#     simulation.reporters.pop()

weights = np.zeros(len(states), np.float)
simulation.reporters.append(atomsmm.ExpandedEnsembleReporter('states.csv', ndisp, 2, states, weights, temp))
simulation.step(nprod)
