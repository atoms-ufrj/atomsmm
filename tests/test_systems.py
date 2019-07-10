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
        assert value/value.unit == pytest.approx(potential[term])


def test_SolvationSystem_with_lj_parameter_scaling():
    system, positions, topology, solute = readSystem('hydroxyethylaminoanthraquinone-in-water')
    solvation_system = atomsmm.SolvationSystem(system, solute, use_softcore=False)
    state = dict(lambda_vdw=0.5, lambda_coul=0.5)
    components = atomsmm.splitPotentialEnergy(solvation_system, topology, positions, **state)
    potential = dict()
    potential['HarmonicBondForce'] = 1815.1848188179738
    potential['HarmonicAngleForce'] = 1111.5544374007236
    potential['PeriodicTorsionForce'] = 1.5998609986459567
    potential['Real-Space'] = 58235.03496195241
    potential['Reciprocal-Space'] = -76436.3982762784
    potential['Total'] = -15273.024197108643
    for term, value in components.items():
        assert value/value.unit == pytest.approx(potential[term])


def test_RESPASystem():
    system, positions, topology, solute = readSystem('hydroxyethylaminoanthraquinone-in-water')
    respa_info = dict(rcutIn=7*unit.angstroms, rswitchIn=5*unit.angstroms)
    solvation_system = atomsmm.SolvationSystem(system, solute)
    respa_system = atomsmm.RESPASystem(solvation_system, *respa_info.values())
    state = dict(lambda_vdw=0.5, lambda_coul=0.5)
    components = atomsmm.splitPotentialEnergy(respa_system, topology, positions, **state)
    potential = dict()
    potential['HarmonicBondForce'] = 1815.1848188179738
    potential['HarmonicAngleForce'] = 1111.5544374007236
    potential['PeriodicTorsionForce'] = 1.5998609986459567
    potential['Real-Space'] = 58161.10011792888
    potential['Reciprocal-Space'] = -76436.3982762784
    potential['CustomNonbondedForce'] = -64.67189605331785
    potential['CustomNonbondedForce(1)'] = -17294.836032921234
    potential['CustomNonbondedForce(2)'] = 17294.836032921194
    potential['CustomBondForce'] = 112.25315524350334
    potential['Total'] = -15299.377781942032
    for term, value in components.items():
        assert value/value.unit == pytest.approx(potential[term])


def test_RESPASystem_with_exception_offsets():
    system, positions, topology, solute = readSystem('hydroxyethylaminoanthraquinone-in-water')
    respa_info = dict(rcutIn=7*unit.angstroms, rswitchIn=5*unit.angstroms)
    solvation_system = atomsmm.SolvationSystem(system, solute)
    nbforce = solvation_system.getForce(atomsmm.findNonbondedForce(solvation_system))
    for index in range(nbforce.getNumExceptions()):
        i, j, chargeprod, sigma, epsilon = nbforce.getExceptionParameters(index)
        nbforce.setExceptionParameters(index, i, j, 0.0, sigma, epsilon)
        nbforce.addExceptionParameterOffset('lambda_coul', index, chargeprod, 0.0, 0.0)
    respa_system = atomsmm.RESPASystem(solvation_system, *respa_info.values())
    state = dict(lambda_vdw=0.5, lambda_coul=0.5)
    components = atomsmm.splitPotentialEnergy(respa_system, topology, positions, **state)
    potential = dict()
    potential['HarmonicBondForce'] = 1815.1848188179738
    potential['HarmonicAngleForce'] = 1111.5544374007236
    potential['PeriodicTorsionForce'] = 1.5998609986459567
    potential['Real-Space'] = 58201.09912379701
    potential['Reciprocal-Space'] = -76436.3982762784
    potential['CustomNonbondedForce'] = -64.67189605331785
    potential['CustomNonbondedForce(1)'] = -17294.836032921234
    potential['CustomNonbondedForce(2)'] = 17294.836032921194
    potential['CustomBondForce'] = 72.25414937535754
    potential['Total'] = -15299.377781942048
    for term, value in components.items():
        assert value/value.unit == pytest.approx(potential[term])


def test_RESPASystem_with_lj_parameter_scaling():
    system, positions, topology, solute = readSystem('hydroxyethylaminoanthraquinone-in-water')
    respa_info = dict(rcutIn=7*unit.angstroms, rswitchIn=5*unit.angstroms)
    solvation_system = atomsmm.SolvationSystem(system, solute, use_softcore=False)
    respa_system = atomsmm.RESPASystem(solvation_system, *respa_info.values())
    state = dict(lambda_vdw=0.5, lambda_coul=0.5)
    components = atomsmm.splitPotentialEnergy(respa_system, topology, positions, **state)
    potential = dict()
    potential['HarmonicBondForce'] = 1815.1848188179738
    potential['HarmonicAngleForce'] = 1111.5544374007236
    potential['PeriodicTorsionForce'] = 1.5998609986459567
    potential['Real-Space'] = 58122.78180670893
    potential['Reciprocal-Space'] = -76436.3982762784
    potential['CustomNonbondedForce'] = -17317.054135213173
    potential['CustomNonbondedForce(1)'] = 17317.054135213126
    potential['CustomBondForce'] = 112.25315524350334
    potential['Total'] = -15273.024197108669
    for term, value in components.items():
        assert value/value.unit == pytest.approx(potential[term])


def test_RESPASystem_with_special_bonds():
    system, positions, topology, solute = readSystem('q-SPC-FW')
    respa_info = dict(rcutIn=7*unit.angstroms, rswitchIn=5*unit.angstroms)
    respa_system = atomsmm.RESPASystem(system, *respa_info.values())
    respa_system.redefine_bond(topology, 'HOH', 'H[1-2]', 'O', 1.05*unit.angstroms)
    respa_system.redefine_angle(topology, 'HOH', 'H[1-2]', 'O', 'H[1-2]', 113*unit.degrees)
    components = atomsmm.splitPotentialEnergy(respa_system, topology, positions)
    potential = dict()
    potential['HarmonicBondForce'] = 3665.684696323676
    potential['HarmonicAngleForce'] = 1811.197218501007
    potential['PeriodicTorsionForce'] = 0.0
    potential['Real-Space'] = 84694.39953220935
    potential['Reciprocal-Space'] = -111582.71281220087
    potential['CustomNonbondedForce'] = -25531.129587235544
    potential['CustomNonbondedForce(1)'] = 25531.129587235544
    potential['CustomBondForce'] = 0.0
    potential['CustomBondForce(1)'] = -1175.253817235862
    potential['CustomAngleForce'] = -305.0221912655623
    potential['Total'] = -22891.707373668243
    for term, value in components.items():
        print(term, value)
        assert value/value.unit == pytest.approx(potential[term])


def test_AlchemicalRespaSystem():
    system, positions, topology, solute = readSystem('phenol-in-water')
    respa_info = dict(rcutIn=7*unit.angstroms, rswitchIn=5*unit.angstroms)
    solvation_system = atomsmm.systems.AlchemicalRespaSystem(
        system,
        *respa_info.values(),
        solute,
        coupling_function='lambda^4*(5-4*lambda)',
    )
    state = {'lambda': 0.5, 'respa_switch': 1}
    components = atomsmm.splitPotentialEnergy(solvation_system, topology, positions, **state)
    for item in components.items():
        print(*item)
    potential = {}
    potential['HarmonicBondForce'] = 2621.3223922886677  # kJ/mol
    potential['HarmonicAngleForce'] = 1525.1006876561419  # kJ/mol
    potential['PeriodicTorsionForce'] = 18.767576693568476  # kJ/mol
    potential['Real-Space'] = 80089.51116719692  # kJ/mol
    potential['Reciprocal-Space'] = -107038.52551657759  # kJ/mol
    potential['CustomNonbondedForce'] = 5037.152491649265  # kJ/mol
    potential['CustomBondForce'] = -53.526446723139806  # kJ/mol
    potential['CustomBondForce(1)'] = -53.374675325650806  # kJ/mol
    potential['CustomCVForce'] = -7.114065227572182  # kJ/mol
    potential['CustomCVForce(1)'] = -6.301336948673654  # kJ/mol
    potential['Total'] = -17866.987725318053  # kJ/mol
    for term, value in components.items():
        assert value/value.unit == pytest.approx(potential[term])


def test_AlchemicalRespaSystem_without_middle_scale():
    system, positions, topology, solute = readSystem('phenol-in-water')
    respa_info = dict(rcutIn=7*unit.angstroms, rswitchIn=5*unit.angstroms)
    solvation_system = atomsmm.systems.AlchemicalRespaSystem(
        system,
        *respa_info.values(),
        solute,
        coupling_function='lambda^4*(5-4*lambda)',
        middle_scale=False,
    )
    state = {'lambda': 0.5}
    components = atomsmm.splitPotentialEnergy(solvation_system, topology, positions, **state)
    for item in components.items():
        print(*item)
    potential = {}
    potential['HarmonicBondForce'] = 2621.3223922886677  # kJ/mol
    potential['HarmonicAngleForce'] = 1525.1006876561419  # kJ/mol
    potential['PeriodicTorsionForce'] = 18.767576693568476  # kJ/mol
    potential['Real-Space'] = 80089.51116719692  # kJ/mol
    potential['Reciprocal-Space'] = -107038.52551657759  # kJ/mol
    potential['CustomBondForce'] = -53.526446723139806  # kJ/mol
    potential['CustomCVForce'] = -7.114065227572182  # kJ/mol
    potential['Total'] = -22844.464204692995  # kJ/mol
    for term, value in components.items():
        assert value/value.unit == pytest.approx(potential[term])


def test_AlchemicalRespaSystem_with_coulomb_scaling():
    system, positions, topology, solute = readSystem('phenol-in-water')
    respa_info = dict(rcutIn=7*unit.angstroms, rswitchIn=5*unit.angstroms)
    solvation_system = atomsmm.systems.AlchemicalRespaSystem(
        system,
        *respa_info.values(),
        solute,
        coupling_function='lambda^4*(5-4*lambda)',
        lambda_coul=0.5,
    )
    state = {'lambda': 0.5, 'respa_switch': 1}
    components = atomsmm.splitPotentialEnergy(solvation_system, topology, positions, **state)
    for item in components.items():
        print(*item)
    potential = {}
    potential['HarmonicBondForce'] = 2621.3223922886677  # kJ/mol
    potential['HarmonicAngleForce'] = 1525.1006876561419  # kJ/mol
    potential['PeriodicTorsionForce'] = 18.767576693568476  # kJ/mol
    potential['Real-Space'] = 80088.80483999196  # kJ/mol
    potential['Reciprocal-Space'] = -107074.49398119976  # kJ/mol
    potential['CustomNonbondedForce'] = 5037.152491649265  # kJ/mol
    potential['CustomBondForce'] = -53.526446723139806  # kJ/mol
    potential['CustomBondForce(1)'] = -53.374675325650806  # kJ/mol
    potential['CustomNonbondedForce(1)'] = -23.447733015522058  # kJ/mol
    potential['CustomCVForce'] = -7.114065227572182  # kJ/mol
    potential['CustomCVForce(1)'] = -6.301336948673654  # kJ/mol
    potential['Total'] = -17927.110250160702  # kJ/mol
    for term, value in components.items():
        assert value/value.unit == pytest.approx(potential[term])
