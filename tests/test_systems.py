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
    print()
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
    print()
    for term, value in components.items():
        print(term, value)
        # assert value/value.unit == pytest.approx(potential[term])


def test_RESPASystem():
    system, positions, topology, solute = readSystem('hydroxyethylaminoanthraquinone-in-water')
    respa_info = dict(rcutIn = 7*unit.angstroms, rswitchIn = 5*unit.angstroms)
    solvation_system = atomsmm.SolvationSystem(system, solute)
    respa_system = atomsmm.RESPASystem(solvation_system, *respa_info.values())
    state = dict(lambda_vdw=0.5, lambda_coul=0.5)
    components = atomsmm.splitPotentialEnergy(respa_system, topology, positions, **state)
    potential = dict()
    potential['HarmonicBondForce'] = 1815.1848188179738
    potential['HarmonicAngleForce'] = 1111.5544374007236
    potential['PeriodicTorsionForce'] = 1.5998609986459567
    potential['Real-Space'] = 58273.35327317236
    potential['Reciprocal-Space'] = -76436.3982762784
    potential['CustomNonbondedForce'] = -64.67189605331785
    potential['Total'] = -15299.377781942014
    print()
    for term, value in components.items():
        print(term, value)
        # assert value/value.unit == pytest.approx(potential[term])

# BEFORE CHANGES:
#
# Solvation system:
#
# HarmonicBondForce 1815.1848188179738 kJ/mol
# HarmonicAngleForce 1111.5544374007236 kJ/mol
# PeriodicTorsionForce 1.5998609986459567 kJ/mol
# Real-Space 58273.35327317236 kJ/mol
# Reciprocal-Space -76436.3982762784 kJ/mol
# CustomNonbondedForce -64.67189605331785 kJ/mol
# Total -15299.377781942014 kJ/mol
#
# Solvation system with RESPA:
#
# HarmonicBondForce 1815.1848188179738 kJ/mol
# HarmonicAngleForce 1111.5544374007236 kJ/mol
# PeriodicTorsionForce 1.5998609986459567 kJ/mol
# Real-Space 58161.10011792888 kJ/mol
# Reciprocal-Space -76436.3982762784 kJ/mol
# CustomNonbondedForce -64.67189605331785 kJ/mol
# CustomNonbondedForce(1) -17164.91406821642 kJ/mol
# CustomNonbondedForce(2) 17164.91406821638 kJ/mol
# CustomBondForce 112.25315524350334 kJ/mol
# CustomNonbondedForce(3) -129.9219647048243 kJ/mol
# CustomNonbondedForce(4) 129.9219647048243 kJ/mol
# Total -15299.377781942032 kJ/mol
