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
                                     nonbondedMethod=openmm.app.PME,
                                     nonbondedCutoff=10*unit.angstroms,
                                     rigidWater=False,
                                     constraints=None,
                                     removeCMMotion=False)
    nbforce = system.getForce(atomsmm.findNonbondedForce(system))
    nbforce.setUseSwitchingFunction(True)
    nbforce.setSwitchingDistance(9*unit.angstroms)
    residues = [atom.residue.name for atom in pdb.topology.atoms()]
    solute_atoms = set(i for (i, name) in enumerate(residues) if name == 'aaa')
    return system, pdb.positions, pdb.topology, solute_atoms


def test_SolvationSystem():
    system, positions, topology, solute = readSystem('hydroxyethylaminoanthraquinone-in-water')
    solvation_system = atomsmm.SolvationSystem(system, solute)
    state = dict(lambda_vdw=0.5, lambda_coul=0.5)
    components = atomsmm.splitPotentialEnergy(solvation_system, topology, positions, **state)
    potential = dict()
    potential['HarmonicBondForce'] = 1815.1848188179738
    potential['HarmonicAngleForce'] = 1111.5544374007236
    potential['PeriodicTorsionForce'] = 1.5998609986459567
    potential['Real-Space'] = 58273.35327317236
    potential['Reciprocal-Space'] = -76436.3982762784
    potential['CustomNonbondedForce'] = -64.67189605331785
    potential['Total'] = -15299.377781942014
    for term, value in components.items():
        print(term, value)
        assert value/value.unit == pytest.approx(potential[term])


def test_SolvationSystemWithRESPA():
    system, positions, topology, solute = readSystem('hydroxyethylaminoanthraquinone-in-water')
    respa_info = dict(rcutIn = 7*unit.angstroms, rswitchIn = 5*unit.angstroms)
    solvation_system = atomsmm.SolvationSystem(system, solute, respa_info)
    state = dict(lambda_vdw=0.5, lambda_coul=0.5)
    components = atomsmm.splitPotentialEnergy(solvation_system, topology, positions, **state)
    potential = dict()
    potential['HarmonicBondForce'] = 1815.1848188179738
    potential['HarmonicAngleForce'] = 1111.5544374007236
    potential['PeriodicTorsionForce'] = 1.5998609986459567
    potential['Real-Space'] = 58273.35327317236
    potential['Reciprocal-Space'] = -76436.3982762784
    potential['CustomNonbondedForce'] = -64.67189605331785
    potential['Total'] = -15299.377781942014
    for term, value in components.items():
        print(term, value)
        # assert value/value.unit == pytest.approx(potential[term])


# def test_RESPASystem():
#     system, positions, topology = readSystem('hydroxyethylaminoanthraquinone-in-water')
#     rcutIn = 7*unit.angstroms
#     rswitchIn = 5*unit.angstroms
#     state = dict(lambda_vdw=0.5, lambda_coul=0.5)
#     respa_system = atomsmm.RESPASystem(system, rcutIn, rswitchIn)
#     components = atomsmm.splitPotentialEnergy(respa_system, topology, positions, **state)
#     potential = dict()
#     potential['HarmonicBondForce'] = 1815.1848188179738
#     potential['HarmonicAngleForce'] = 1111.5544374007236
#     potential['PeriodicTorsionForce'] = 1.5998609986459567
#     potential['Real-Space'] = 58273.35327317236
#     potential['Reciprocal-Space'] = -76436.3982762784
#     potential['CustomNonbondedForce'] = -64.67189605331785
#     potential['Total'] = -15299.377781942014
#     for term, value in components.items():
#         print(term, value)
#         # assert value/value.unit == pytest.approx(potential[term])
