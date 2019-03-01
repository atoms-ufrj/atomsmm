"""
.. module:: computers
   :platform: Unix, Windows
   :synopsis: a module for defining computers, which are subclasses of OpenMM Context_ class.

.. moduleauthor:: Charlles R. A. Abreu <abreu@eq.ufrj.br>

.. _Context: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.Context.html

"""

import itertools

import numpy as np
from scipy import sparse
from simtk import openmm
from simtk import unit

import atomsmm


class _MoleculeTotalizer(object):
    def __init__(self, context, topology):
        molecules = context.getMolecules()
        atoms = list(itertools.chain.from_iterable(molecules))
        nmols = self.nmols = len(molecules)
        natoms = self.natoms = len(atoms)
        mol = sum([[i]*len(molecule) for i, molecule in enumerate(molecules)], [])

        def sparseMatrix(data):
            return sparse.csr_matrix((data, (mol, atoms)), shape=(nmols, natoms))

        selection = self.selection = sparseMatrix(np.ones(natoms, np.int))
        system = context.getSystem()
        mass = np.array([system.getParticleMass(i).value_in_unit(unit.dalton) for i in range(natoms)])
        molMass = self.molMass = selection.dot(mass)
        total = selection.T.dot(molMass)
        self.massFrac = sparseMatrix(mass/total)

        atomResidues = {}
        for atom in topology.atoms():
            atomResidues[int(atom.index)-1] = atom.residue.name
        self.residues = [atomResidues[item[0]] for item in molecules]


class VirialComputer(openmm.Context):
    def __init__(self, system, topology, platform, properties=dict(), **kwargs):
        self._system = atomsmm.ComputingSystem(system, **kwargs)
        super().__init__(self._system, openmm.CustomIntegrator(0), platform, properties)
        self._mols = _MoleculeTotalizer(self, topology)
        self._make_obsolete()

    def _get_forces(self, groups):
        return self.getState(getForces=True, groups=groups).getForces(asNumpy=True)

    def _get_positions(self):
        return self.getState(getPositions=True).getPositions(asNumpy=True)

    def _get_potential(self, groups):
        return self.getState(getEnergy=True, groups=groups).getPotentialEnergy()

    def _get_velocities(self):
        return self.getState(getVelocities=True).getVelocities(asNumpy=True)

    def _make_obsolete(self):
        self._bond_virial = None
        self._coulomb_virial = None
        self._dispersion_virial = None
        self._molecular_virial = None
        self._molecular_kinetic_energy = None

    def get_atomic_virial(self):
        return self.get_bond_virial() + self.get_coulomb_virial() + self.get_dispersion_virial()

    def get_bond_virial(self):
        if self._bond_virial is None:
            self._bond_virial = self._get_potential(self._system._bonded)
        return self._bond_virial

    def get_molecular_kinetic_energy(self):
        if self._molecular_kinetic_energy is None:
            velocities = self._get_velocities().value_in_unit(unit.nanometers/unit.picosecond)
            vcm = self._mols.massFrac.dot(velocities)
            Kcm = self._mols.molMass*np.sum(vcm**2, axis=1)
            self._molecular_kinetic_energy = np.sum(Kcm)*unit.kilojoules_per_mole
        return self._molecular_kinetic_energy

    def get_coulomb_virial(self):
        if self._coulomb_virial is None:
            self._coulomb_virial = self._get_potential(self._system._coulomb)
        return self._coulomb_virial

    def get_dispersion_virial(self):
        if self._dispersion_virial is None:
            self._dispersion_virial = self._get_potential(self._system._dispersion)
        return self._dispersion_virial

    def get_molecular_virial(self, forces):
        if self._molecular_virial is None:
            others = self._get_forces(self._system._others)
            f = (forces - others).value_in_unit(unit.kilojoules_per_mole/unit.nanometers)
            r = self._get_positions().value_in_unit(unit.nanometers)
            fcm = self._mols.selection.dot(f)
            rcm = self._mols.massFrac.dot(r)
            W = self.get_atomic_virial().value_in_unit(unit.kilojoules_per_mole)
            self._molecular_virial = (W + np.sum(rcm*fcm) - np.sum(r*f))*unit.kilojoules_per_mole
        return self._molecular_virial

    def get_volume(self):
        box = self.getState().getPeriodicBoxVectors()
        return box[0][0]*box[1][1]*box[2][2]

    def import_configuration(self, state):
        self.setPeriodicBoxVectors(*state.getPeriodicBoxVectors())
        self.setPositions(state.getPositions())
        self.setVelocities(state.getVelocities())
        self._make_obsolete()
