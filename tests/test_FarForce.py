from __future__ import print_function

import pytest
from simtk import openmm
from simtk import unit
from simtk.openmm import app

import atomsmm


def execute(OuterForceType, shifted):
    rswitch_inner = 6.5*unit.angstroms
    rcut_inner = 7.0*unit.angstroms
    rswitch = 9.5*unit.angstroms
    rcut = 10*unit.angstroms
    case = 'tests/data/q-SPC-FW'
    pdb = app.PDBFile(case + '.pdb')
    forcefield = app.ForceField(case + '.xml')

    system = forcefield.createSystem(pdb.topology)
    nbforce = atomsmm.hijackForce(system, atomsmm.findNonbondedForce(system))
    innerforce = atomsmm.NearNonbondedForce(rcut_inner, rswitch_inner, shifted).setForceGroup(1)
    innerforce.importFrom(nbforce).addTo(system)
    outerforce = OuterForceType(innerforce, rcut, rswitch).setForceGroup(2)
    outerforce.importFrom(nbforce).addTo(system)
    potential = atomsmm.splitPotentialEnergy(system, pdb.topology, pdb.positions)["Total"]

    refsys = forcefield.createSystem(pdb.topology,
                                     nonbondedMethod=openmm.app.PME,
                                     nonbondedCutoff=rcut,
                                     removeCMMotion=True)
    force = refsys.getForce(refsys.getNumForces()-2)
    force.setUseSwitchingFunction(True)
    force.setSwitchingDistance(rswitch)
    force.setEwaldErrorTolerance(1E-5)
    refpot = atomsmm.splitPotentialEnergy(refsys, pdb.topology, pdb.positions)["Total"]

    assert potential/potential.unit == pytest.approx(refpot/refpot.unit)


def test_unshifted():
    execute(atomsmm.FarNonbondedForce, False)


def test_shifted():
    execute(atomsmm.FarNonbondedForce, True)
