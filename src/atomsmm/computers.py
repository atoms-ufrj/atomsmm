"""
.. module:: computers
   :platform: Unix, Windows
   :synopsis: a module for defining computers, which are subclasses of OpenMM Context_ class.

.. moduleauthor:: Charlles R. A. Abreu <abreu@eq.ufrj.br>

.. _Context: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.Context.html

"""

from simtk import openmm

import atomsmm


class VirialComputer(openmm.Context):
    def __init__(self, system, positions, platform, properties=dict(), **kwargs):
        integrator = openmm.CustomIntegrator(0)
        self._system = atomsmm.ComputingSystem(system, **kwargs)
        self._positions = positions
        self._box = system.getDefaultPeriodicBoxVectors()
        super().__init__(self._system, integrator, platform, properties)
        self.setPositions(positions)
        self._bond_virial = None
        self._coulomb_virial = None
        self._dispersion_virial = None

    def _potential(self, groups):
        groupState = self.getState(getEnergy=True, groups=groups)
        return groupState.getPotentialEnergy()

    def get_atomic_virial(self):
        return self.get_bond_virial() + self.get_coulomb_virial() + self.get_dispersion_virial()

    def get_bond_virial(self):
        if self._bond_virial is None:
            self._bond_virial = self._potential(self._system._bonded)
        return self._bond_virial

    def get_coulomb_virial(self):
        if self._coulomb_virial is None:
            self._coulomb_virial = self._potential(self._system._coulomb)
        return self._coulomb_virial

    def get_dispersion_virial(self):
        if self._dispersion_virial is None:
            self._dispersion_virial = self._potential(self._system._dispersion)
        return self._dispersion_virial

    def get_volume(self):
        return self._box[0][0]*self._box[1][1]*self._box[2][2]

    def get_positions(self):
        return self._positions

    def import_configuration(self, state):
        self._box = state.getPeriodicBoxVectors()
        self._positions = state.getPositions(asNumpy=True)
        self.setPeriodicBoxVectors(*self._box)
        self.setPositions(self._positions)
        self._bond_virial = None
        self._coulomb_virial = None
        self._dispersion_virial = None
