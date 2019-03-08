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
        self.mass = np.array([system.getParticleMass(i).value_in_unit(unit.dalton) for i in range(natoms)])
        molMass = self.molMass = selection.dot(self.mass)
        total = selection.T.dot(molMass)
        self.massFrac = sparseMatrix(self.mass/total)

        atomResidues = {}
        for atom in topology.atoms():
            atomResidues[int(atom.index)-1] = atom.residue.name
        self.residues = [atomResidues[item[0]] for item in molecules]


class PressureComputer(openmm.Context):
    """
    An OpenMM Context_ extension aimed at computing properties of a system related to isotropic
    volume variations.

    Parameters
    ----------
        system : openmm.System
            The system...
        topology : openmm.app.Topology
            The topology...
        platform : openmm.Platform
            The platform...
        properties : dict(), optional, default=dict()
            The properties...
        temperature : unit.Quantity, optional, default=None
            The bath temperature used to compute pressures using the equipartition expectations of
            kinetic energies. It this is `None`, then the instantaneous kinetic energies will be
            employed.

    """
    def __init__(self, system, topology, platform, properties=dict(), temperature=None):
        self._system = atomsmm.ComputingSystem(system)
        super().__init__(self._system, openmm.CustomIntegrator(0), platform, properties)
        self._mols = _MoleculeTotalizer(self, topology)
        self._kT = None if temperature is None else unit.MOLAR_GAS_CONSTANT_R*temperature
        self._make_obsolete()

    def _get_forces(self, groups):
        return self.getState(getForces=True, groups=groups).getForces(asNumpy=True)

    def _get_positions(self):
        return self.getState(getPositions=True).getPositions(asNumpy=True)

    def _get_potential(self, groups):
        return self.getState(getEnergy=True, groups=groups).getPotentialEnergy()

    def _get_velocities(self):
        return self.getState(getVelocities=True).getVelocities(asNumpy=True)

    def _get_volume(self):
        box = self.getState().getPeriodicBoxVectors()
        return box[0][0]*box[1][1]*box[2][2]*unit.AVOGADRO_CONSTANT_NA

    def _make_obsolete(self):
        self._bond_virial = None
        self._coulomb_virial = None
        self._dispersion_virial = None
        self._molecular_virial = None
        self._molecular_kinetic_energy = None

    def get_atomic_pressure(self):
        """
        Returns the unconstrained atomic pressure of a system:

        .. math::
            P = \\frac{2 K + W}{3 V},

        where :math:`W` is the unconstrained atomic virial (see :func:`get_atomic_virial`),
        :math:`K` is the total kinetic energy of all atoms, and :math:`V` is the box volume. If
        keyword `temperature` was employed in the :class:`PressureComputer` creation, then the
        instantaneous kinetic energy is replaced by its equipartition-theorem average
        :math:`\\left\\langle K \\right\\rangle = 3 N_\\mathrm{atoms} k_B T/2`, where :math:`T`
        is the heat-bath temperature, thus making :math:`P` independent of the atomic velocities.

        .. warning::
            The resulting pressure should not be used to compute the thermodynamic pressure of a
            system with constraints. For this, one can use :func:`get_molecular_pressure` instead.

        """
        if self._kT is None:
            velocities = self._get_velocities().value_in_unit(unit.nanometers/unit.picosecond)
            mvv = self._mols.mass*np.sum(velocities**2, axis=1)
            dNkT = np.sum(mvv)*unit.kilojoules_per_mole
        else:
            dNkT = 3*self._mols.natoms*self._kT
        pressure = (dNkT + self.get_atomic_virial())/(3*self._get_volume())
        return pressure.in_units_of(unit.atmospheres)

    def get_atomic_virial(self):
        """
        Returns the unconstrained atomic virial of the system.

        Considering full scaling of atomic coordinates in a box volume change (i.e. without any
        distance constraints), the internal virial of the system is given by

        .. math::
            W = -\\sum_{i,j} r_{ij} E^\\prime(r_{ij}),

        where :math:`E^\\prime(r)` is the derivative of the interaction potential as a function of
        the distance between two atoms. Such interaction includes van der Waals, Coulomb, and
        bond-stretching contributions. Angles and dihedrals are not considered because they are
        invariant to full atomic coordinate scaling.

        .. warning::
            The resulting virial should not be used to compute the thermodynamic pressure of a
            system with constraints. For this, one can use :func:`get_molecular_virial` instead.

        """
        return self.get_bond_virial() + self.get_coulomb_virial() + self.get_dispersion_virial()

    def get_bond_virial(self):
        """
        Returns the bond-stretching contribution to the atomic virial.

        """
        if self._bond_virial is None:
            self._bond_virial = self._get_potential(self._system._bonded)
        return self._bond_virial

    def get_coulomb_virial(self):
        """
        Returns the electrostatic (Coulomb) contribution to the atomic virial.

        """
        if self._coulomb_virial is None:
            self._coulomb_virial = self._get_potential(self._system._coulomb)
        return self._coulomb_virial

    def get_dispersion_virial(self):
        """
        Returns the dispersion (van der Waals) contribution to the atomic virial.

        """
        if self._dispersion_virial is None:
            self._dispersion_virial = self._get_potential(self._system._dispersion)
        return self._dispersion_virial

    def get_molecular_kinetic_energy(self):
        if self._molecular_kinetic_energy is None:
            velocities = self._get_velocities().value_in_unit(unit.nanometers/unit.picosecond)
            vcm = self._mols.massFrac.dot(velocities)
            mvv = self._mols.molMass*np.sum(vcm**2, axis=1)
            self._molecular_kinetic_energy = 0.5*np.sum(mvv)*unit.kilojoules_per_mole
        return self._molecular_kinetic_energy

    def get_molecular_pressure(self, forces):
        """
        Returns the molecular pressure of a system:

        .. math::
            P = \\frac{2 K_\\mathrm{mol} + W_\\mathrm{mol}}{3 V},

        where :math:`W_\\mathrm{mol}` is the molecular virial of the system (see
        :func:`get_molecular_virial`), :math:`K_\\mathrm{mol}` is the center-of-mass kinetic energy
        summed for all molecules, and :math:`V` is the box volume. If keyword `temperature` is
        was employed in the :class:`PressureComputer` creation, then the moleculer kinetic energy is
        replaced by its equipartition-theorem average
        :math:`\\left\\langle K_\\mathrm{mol} \\right\\rangle = 3 N_\\mathrm{mols} k_B T/2`,
        where :math:`T` is the heat-bath temperature.

        Parameter
        ---------
            forces : vector<openmm.Vec3>
                A vector whose length equals the number of particles in the System. The i-th element
                contains the force on the i-th particle.

        """
        if self._kT is None:
            dNkT = 2.0*self.get_molecular_kinetic_energy()
        else:
            dNkT = 3*self._mols.nmols*self._kT
        pressure = (dNkT + self.get_molecular_virial(forces))/(3*self._get_volume())
        return pressure.in_units_of(unit.atmospheres)

    def get_molecular_virial(self, forces):
        """
        Returns the molecular virial of a system.

        To compute the molecular virial, only the center-of-mass coordinates of the molecules are
        considered to scale in a box volume change, while the internal molecular structure keeps
        rigid. The molecular virial is computed from the nonbonded part of the atomic virial by
        using the formulation of Ref. :cite:`Hunenberger_2002`:

        .. math::
            W_\\mathrm{mol} = W -
                \\sum_{i} (\\mathbf{r}_i -\\mathbf{r}_i^\\mathrm{cm}) \\cdot \\mathbf{F}_i,

        where :math:`\\mathbf{r}_i` is the coordinate of atom i, :math:`\\mathbf{F}_i` is the
        resultant pairwise force acting on it, and :math:`\\mathbf{r}_i^\\mathrm{cm}` is the
        center-of-mass coordinate of the molecule to which it belongs.

        Parameter
        ---------
            forces : vector<openmm.Vec3>
                A vector whose length equals the number of particles in the System. The i-th element
                contains the force on the i-th particle.

        """
        f = forces.value_in_unit(unit.kilojoules_per_mole/unit.nanometers)
        r = self._get_positions().value_in_unit(unit.nanometers)
        fcm = self._mols.selection.dot(f)
        rcm = self._mols.massFrac.dot(r)
        W = self.get_atomic_virial().value_in_unit(unit.kilojoules_per_mole)
        return (W + np.sum(rcm*fcm) - np.sum(r*f))*unit.kilojoules_per_mole

    def import_configuration(self, state):
        self.setPeriodicBoxVectors(*state.getPeriodicBoxVectors())
        self.setPositions(state.getPositions())
        self.setVelocities(state.getVelocities())
        self._make_obsolete()
