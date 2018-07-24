from __future__ import print_function

import pytest
from simtk import openmm
from simtk import unit
from simtk.openmm import app

import atomsmm


def execute(shifted):
    rswitch_inner = 6.5*unit.angstroms
    rcut_inner = 7.0*unit.angstroms
    rswitch = 9.5*unit.angstroms
    rcut = 10*unit.angstroms
    case = 'tests/data/q-SPC-FW'
    pdb = app.PDBFile(case + '.pdb')
    forcefield = app.ForceField(case + '.xml')

    system = forcefield.createSystem(pdb.topology)
    nbforce = atomsmm.hijackNonbondedForce(system)
    exceptions = atomsmm.NonbondedExceptionForce()
    exceptions.importFrom(nbforce).addTo(system)
    innerforce = atomsmm.InnerRespaForce(rcut_inner, rswitch_inner, shifted).setForceGroup(1)
    innerforce.importFrom(nbforce).addTo(system)
    outerforce = atomsmm.OuterRespaForce(innerforce, rcut, rswitch).setForceGroup(2)
    outerforce.importFrom(nbforce).addTo(system)
    potential = atomsmm.splitPotentialEnergy(system, pdb.topology, pdb.positions)

    refsys = forcefield.createSystem(pdb.topology,
                                     nonbondedMethod=openmm.app.PME,
                                     nonbondedCutoff=rcut,
                                     removeCMMotion=True)
    force = refsys.getForce(refsys.getNumForces()-2)
    force.setUseSwitchingFunction(True)
    force.setSwitchingDistance(rswitch)
    refpot = atomsmm.splitPotentialEnergy(refsys, pdb.topology, pdb.positions)

    E = potential["Total"]
    refE = refpot["Total"]
    assert E/E.unit == pytest.approx(refE/refE.unit)

    E = potential["CustomBondForce"] + potential["NonbondedForce"]
    refE = refpot["NonbondedForce"]
    assert E/E.unit == pytest.approx(refE/refE.unit)


def test_unshifted():
    execute(False)
