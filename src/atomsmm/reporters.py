"""
.. module:: reporters
   :platform: Unix, Windows
   :synopsis: a module for defining OpenMM reporter classes.

.. moduleauthor:: Charlles R. A. Abreu <abreu@eq.ufrj.br>

.. _pandas.DataFrame: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html
.. _StateDataReporter: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.app.statedatareporter.StateDataReporter.html

"""

import sys

import numpy as np
import pandas as pd
from simtk import openmm
from simtk import unit
from simtk.openmm import app

from .computers import PressureComputer
from .computers import _MoleculeTotalizer
from .utils import InputError


class _MultiStream:
    def __init__(self, outputs):
        self._outputs = list()
        for output in outputs:
            self._outputs.append(open(output, 'w') if isinstance(output, str) else output)

    def __del__(self):
        for output in self._outputs:
            if output != sys.stdout and output != sys.stderr:
                output.close()

    def write(self, message):
        for output in self._outputs:
            output.write(message)

    def flush(self):
        for output in self._outputs:
            output.flush()


class _AtomsMM_Reporter():
    """
    Base class for reporters.

    """
    def __init__(self, file, reportInterval, **kwargs):
        self._reportInterval = reportInterval
        self._requiresInitialization = True
        self._needsPositions = False
        self._needsVelocities = False
        self._needsForces = False
        self._needEnergy = False
        extraFile = kwargs.pop('extraFile', None)
        if extraFile is None:
            self._out = open(file, 'w') if isinstance(file, str) else file
        else:
            self._out = _MultiStream([file, extraFile])
        self._separator = kwargs.pop('separator', ',')

    def _initialize(self, simulation, state):
        pass

    def _generateReport(self, simulation, state):
        pass

    def describeNextReport(self, simulation):
        """
        Get information about the next report this object will generate.

        Parameters
        ----------
            simulation : Simulation
                The Simulation to generate a report for

        Returns
        -------
            tuple
                A five element tuple. The first element is the number of steps
                until the next report. The remaining elements specify whether
                that report will require positions, velocities, forces, and
                energies respectively.

        """
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return (steps, self._needsPositions, self._needsVelocities, self._needsForces, self._needEnergy)

    def report(self, simulation, state):
        """
        Generate a report.

        Parameters
        ----------
            simulation : Simulation
                The Simulation to generate a report for
            state : State
                The current state of the simulation

        """
        if self._requiresInitialization:
            self._initialize(simulation, state)
            self._requiresInitialization = False
        self._generateReport(simulation, state)


class ExtendedStateDataReporter(app.StateDataReporter):
    """
    An extension of OpenMM's StateDataReporter_ class, which outputs information about a simulation,
    such as energy and temperature, to a file.

    All original functionalities of StateDataReporter_ are preserved and the following ones are
    included:

    1. Report the Coulomb contribution of the potential energy (keyword: `coulombEnergy`):

        This contribution includes both real- and reciprocal-space terms.

    2. Report the atomic virial of a fully-flexible system (keyword: `atomicVirial`):

        Considering full scaling of atomic coordinates in a box volume change (i.e. without any
        distance constraints), the internal virial of the system is given by

        .. math::
            W = -\\sum_{i,j} r_{ij} E^\\prime(r_{ij}),

        where :math:`E^\\prime(r)` is the derivative of the pairwise interaction potential as a
        function of the distance between to atoms. Such interaction includes van der Waals, Coulomb,
        and bond-stretching contributions. Bond-bending and dihedral angles are not considered
        because they are invariant to full volume-scaling of atomic coordinates.

    3. Report the nonbonded contribution of the atomic virial (keyword: `nonbondedVirial`):

        The nonbonded virial is given by

        .. math::
            W_\\mathrm{nb} = -\\sum_{i,j} r_{ij} E_\\mathrm{nb}^\\prime(r_{ij}),

        where :math:`E_\\mathrm{nb}^\\prime(r)` is the derivative of the nonbonded pairwise
        potential, which comprises van der Waals and Coulomb interactions only.

    4. Report the atomic pressure of a fully-flexible system (keyword: `atomicPressure`):

        .. math::
            P = \\frac{2 K + W}{3 V},

        where :math:`K` is the kinetic energy sum for all atoms in the system. If keyword
        `bathTemperature` is employed (see below), the instantaneous kinetic energy is substituted
        by its equipartition-theorem average
        :math:`\\left\\langle K \\right\\rangle = 3 N_\\mathrm{atoms} k_B T/2`,
        where :math:`T` is the heat-bath temperature.

    5. Report the molecular virial of a system (keyword: `molecularVirial`):

        To compute the molecular virial, only the center-of-mass coordinates of the molecules are
        considered to scale in a box volume change, while the internal molecular structure is kept
        unaltered. The molecular virial is computed from the nonbonded part of the atomic virial by
        using the formulation of Ref. :cite:`Hunenberger_2002`:

        .. math::
            W_\\mathrm{mol} = W - \\sum_{i} (\\mathbf{r}_i - \\mathbf{r}_i^\\mathrm{cm}) \\cdot \\mathbf{F}_i,

        where :math:`\\mathbf{r}_i` is the coordinate of atom i, :math:`\\mathbf{F}_i` is the
        resultant pairwise force acting on it (excluding bond-bending and dihedral angles), and
        :math:`\\mathbf{r}_i^\\mathrm{cm}` is the center-of-mass coordinate of its containing
        molecule.

    6. Report the molecular pressure of a system (keyword: `molecularPressure`):

        .. math::
            P = \\frac{2 K_\\mathrm{mol} + W_\\mathrm{mol}}{3 V},

        where :math:`K_\\mathrm{mol}` is the center-of-mass kinetic energy summed for all molecules
        in the system. If keyword `bathTemperature` is employed (see below), the instantaneous
        kinetic energy is substituted by its equipartition-theorem average
        :math:`\\left\\langle K_\\mathrm{mol} \\right\\rangle = 3 N_\\mathrm{mols} k_B T/2`,
        where :math:`T` is the heat-bath temperature.

    7. Report the center-of-mass kinetic energy (keyword: `molecularKineticEnergy`):

        .. math::
            K_\\mathrm{mol} = \\frac{1}{2} \\sum_{i=1}^{N_\\mathrm{mol}} M_i v_{\\mathrm{cm}, i}^2,

        where :math:`N_\\mathrm{mol}` is the number of molecules in the system, :math:`M_i` is the
        total mass of molecule `i`, and :math:`v_{\\mathrm{cm}, i}` is the center-of-mass velocity
        of molecule `i`.

    8. Report potential energies at multiple states (keyword: `globalParameterStates`):

        Computes and reports the potential energy of the system at a number of provided global
        parameter states.

    9. Allow specification of an extra file for reporting (keyword: `extraFile`).

        This can be used for replicating a report simultaneously to `sys.stdout` and to a file
        using a unique reporter.

    Keyword Args
    ------------
        coulombEnergy : bool, optional, default=False
            Whether to write the Coulomb contribution of the potential energy to the file.
        atomicVirial : bool, optional, default=False
            Whether to write the total atomic virial to the file.
        nonbondedVirial : bool, optional, default=False
            Whether to write the nonbonded contribution to the atomic virial to the file.
        atomicPressure : bool, optional, default=False
            Whether to write the internal atomic pressure to the file.
        molecularVirial : bool, optional, default=False
            Whether to write the molecular virial to the file.
        molecularPressure : bool, optional, default=False
            Whether to write the internal molecular pressure to the file.
        molecularKineticEnergy : bool, optional, default=False
            Whether to write the molecular center-of-mass kinetic energy to the file.
        globalParameterStates : pandas.DataFrame_, optional, default=None
            A DataFrame containing context global parameters (column names) and sets of values
            thereof. If it is provided, then the potential energy will be reported for every state
            these parameters define.
        globalParameters : list(str), optional, default=None
            A list of global parameter names. If it is provided, then the values of these parameters
            will be reported.
        pressureComputer : :class:`~atomsmm.computers.PressureComputer`, optional, default=None
            A computer designed to determine pressures and virials. This is mandatory if any keyword
            related to virial or pressure is set as `True`.
        extraFile : str or file, optional, default=None
            Extra file to write to, specified as a file name or a file object.

    """
    def __init__(self, file, reportInterval, **kwargs):
        self._coulombEnergy = kwargs.pop('coulombEnergy', False)
        self._atomicVirial = kwargs.pop('atomicVirial', False)
        self._nonbondedVirial = kwargs.pop('nonbondedVirial', False)
        self._atomicPressure = kwargs.pop('atomicPressure', False)
        self._molecularVirial = kwargs.pop('molecularVirial', False)
        self._molecularPressure = kwargs.pop('molecularPressure', False)
        self._molecularKineticEnergy = kwargs.pop('molecularKineticEnergy', False)
        self._globalParameterStates = kwargs.pop('globalParameterStates', None)
        self._globalParameters = kwargs.pop('globalParameters', None)
        self._pressureComputer = kwargs.pop('pressureComputer', None)
        extra = kwargs.pop('extraFile', None)
        if extra is None:
            super().__init__(file, reportInterval, **kwargs)
        else:
            super().__init__(_MultiStream([file, extra]), reportInterval, **kwargs)
        self._computing = any([self._coulombEnergy,
                               self._atomicVirial,
                               self._nonbondedVirial,
                               self._atomicPressure,
                               self._molecularVirial,
                               self._molecularPressure,
                               self._molecularKineticEnergy])
        if self._computing:
            if self._pressureComputer is not None and not isinstance(self._pressureComputer, PressureComputer):
                raise InputError('keyword "pressureComputer" requires a PressureComputer instance')
            self._needsPositions = True
            self._needsForces = any([self._needsForces,
                                     self._molecularVirial,
                                     self._molecularPressure])
            self._needsVelocities = any([self._needsVelocities,
                                         self._molecularPressure,
                                         self._atomicPressure,
                                         self._molecularKineticEnergy])
        self._backSteps = -sum([self._speed, self._elapsedTime, self._remainingTime])

    def _add_item(self, lst, item):
        if self._backSteps == 0:
            lst.append(item)
        else:
            lst.insert(self._backSteps, item)

    def _constructHeaders(self):
        headers = super()._constructHeaders()
        if self._coulombEnergy:
            self._add_item(headers, 'Coulomb Energy (kJ/mole)')
        if self._atomicVirial:
            self._add_item(headers, 'Atomic Virial (kJ/mole)')
        if self._nonbondedVirial:
            self._add_item(headers, 'Nonbonded Virial (kJ/mole)')
        if self._atomicPressure:
            self._add_item(headers, 'Atomic Pressure (atm)')
        if self._molecularVirial:
            self._add_item(headers, 'Molecular Virial (kJ/mole)')
        if self._molecularPressure:
            self._add_item(headers, 'Molecular Pressure (atm)')
        if self._molecularKineticEnergy:
            self._add_item(headers, 'Molecular Kinetic Energy (kJ/mole)')
        if self._globalParameterStates is not None:
            for index in self._globalParameterStates.index:
                self._add_item(headers, 'Energy[{}] (kJ/mole)'.format(index))
        if self._globalParameters is not None:
            for name in self._globalParameters:
                self._add_item(headers, name)
        return headers

    def _constructReportValues(self, simulation, state):
        values = super()._constructReportValues(simulation, state)
        if self._computing:
            computer = self._pressureComputer
            computer.import_configuration(state)
            atomicVirial = computer.get_atomic_virial().value_in_unit(unit.kilojoules_per_mole)
            if self._coulombEnergy:
                coulombVirial = computer.get_coulomb_virial()
                self._add_item(values, coulombVirial.value_in_unit(unit.kilojoules_per_mole))
            if self._atomicVirial:
                self._add_item(values, atomicVirial)
            if self._nonbondedVirial:
                nonbondedVirial = computer.get_dispersion_virial() + computer.get_coulomb_virial()
                self._add_item(values, nonbondedVirial.value_in_unit(unit.kilojoules_per_mole))
            if self._atomicPressure:
                atomicPressure = computer.get_atomic_pressure()
                self._add_item(values, atomicPressure.value_in_unit(unit.atmospheres))
            if self._molecularVirial or self._molecularPressure:
                forces = state.getForces(asNumpy=True)
                if self._molecularVirial:
                    molecularVirial = computer.get_molecular_virial(forces)
                    self._add_item(values, molecularVirial.value_in_unit(unit.kilojoules_per_mole))
                if self._molecularPressure:
                    molecularPressure = computer.get_molecular_pressure(forces)
                    self._add_item(values, molecularPressure.value_in_unit(unit.atmospheres))
            if self._molecularKineticEnergy:
                molKinEng = computer.get_molecular_kinetic_energy()
                self._add_item(values, molKinEng.value_in_unit(unit.kilojoules_per_mole))

        if self._globalParameterStates is not None:
            original = dict()
            for name in self._globalParameterStates.columns:
                original[name] = simulation.context.getParameter(name)
            latest = original.copy()
            for index, row in self._globalParameterStates.iterrows():
                for name, value in row.items():
                    if value != latest[name]:
                        simulation.context.setParameter(name, value)
                        latest[name] = value
                energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
                self._add_item(values, energy.value_in_unit(unit.kilojoules_per_mole))
            for name, value in original.items():
                if value != latest[name]:
                    simulation.context.setParameter(name, value)

        if self._globalParameters is not None:
            for name in self._globalParameters:
                self._add_item(values, simulation.context.getParameter(name))

        return values


class CenterOfMassReporter(_AtomsMM_Reporter):
    """
    Outputs a series of frames containing the center-of-mass coordinates of all molecules from a
    Simulation to an XYZ-format file.

    To use it, create a CenterOfMassReporter, then add it to the Simulation's list of reporters.

    """
    def __init__(self, file, reportInterval, **kwargs):
        super().__init__(file, reportInterval, **kwargs)
        self._needsPositions = True

    def _initialize(self, simulation, state):
        self._mols = _MoleculeTotalizer(simulation.context, simulation.topology)

    def _generateReport(self, simulation, state):
        positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometers)
        cmPositions = self._mols.massFrac.dot(positions)
        print(self._mols.nmols, file=self._out)
        pd.DataFrame(index=self._mols.residues, data=cmPositions).to_csv(self._out, sep='\t')


class XYZReporter(_AtomsMM_Reporter):
    def __init__(self, file, reportInterval, **kwargs):
        self._atoms = kwargs.get('atoms', 'all')
        self._output = kwargs.get('output', 'positions')
        self._groups = kwargs.get('groups', None)
        if self._output not in ['positions', 'velocities', 'forces']:
            raise InputError('Unrecognizable keyword argument value')
        super().__init__(file, reportInterval, **kwargs)
        self._needsPositions = self._output == 'positions'
        self._needsVelocities = self._output == 'velocities'
        self._needsForces = self._output == 'forces'

    def _initialize(self, simulation, state):
        if self._atoms == 'all':
            self._atoms = range(simulation.topology.getNumAtoms())
        symbol = [atom.element.symbol for atom in simulation.topology.atoms()]
        self._symbols = [symbol[i] for i in self._atoms]
        self._N = len(self._atoms)

    def _generateReport(self, simulation, state):
        if self._output == 'positions':
            values = state.getPositions(asNumpy=True).value_in_unit(unit.angstroms)
        elif self._output == 'velocities':
            values = state.getVelocities(asNumpy=True).value_in_unit(unit.angstroms/unit.picoseconds)
        elif self._groups is None:
            values = state.getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole/unit.nanometers)
        else:
            new_state = simulation.context.getState(getForces=True, groups=self._groups)
            values = new_state.getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole/unit.nanometers)

        print(self._N, file=self._out)
        print('# timestep: {}'.format(simulation.currentStep), file=self._out)
        for symbol, atom in zip(self._symbols, self._atoms):
            xyz = '{:.4f} {:.4f} {:.4f}'.format(*values[atom, :])
            print(symbol, xyz, file=self._out)


class CustomIntegratorReporter(_AtomsMM_Reporter):
    """
    Outputs global and per-DoF variables of a CustomIntegrator instance.

    Keyword Args
    ------------
        describeOnly : bool, optional, default=True
            Whether to output only descriptive statistics that summarize the activated per-Dof
            variables.

    """
    def __init__(self, file, reportInterval, **kwargs):
        super().__init__(file, reportInterval, **kwargs)
        self._describeOnly = kwargs.pop('describeOnly', True)
        self._variables = []
        for key, value in kwargs.items():
            if value is True:
                self._variables.append(key)
        if not self._variables:
            raise InputError("No global or perDof variables have been passed")

    def _initialize(self, simulation, state):
        integrator = self._integrator = simulation.integrator
        if not isinstance(integrator, openmm.CustomIntegrator):
            raise Exception("simulation.integrator is not a CustomIntegrator")
        self._globals = {}
        for index in range(integrator.getNumGlobalVariables()):
            variable = integrator.getGlobalVariableName(index)
            if variable in self._variables:
                self._globals[variable] = index
        self._perDof = {}
        for index in range(integrator.getNumPerDofVariables()):
            variable = integrator.getPerDofVariableName(index)
            if variable in self._variables:
                self._perDof[variable] = index
        if set(self._variables) != set(self._globals) | set(self._perDof):
            raise InputError("Unknown variables have been passed")

    def _generateReport(self, simulation, state):
        for variable, index in self._globals.items():
            value = self._integrator.getGlobalVariable(index)
            print('{}\n{}'.format(variable, value), file=self._out)

        for variable, index in self._perDof.items():
            values = self._integrator.getPerDofVariable(index)
            titles = ['{}.{}'.format(variable, dir) for dir in ['x', 'y', 'z']]
            df = pd.DataFrame(data=np.array(values), columns=titles)
            if self._describeOnly:
                print(df.describe(), file=self._out)
            else:
                df.to_csv(self._out, sep='\t')


class ExpandedEnsembleReporter(_AtomsMM_Reporter):
    """
    Performs an Expanded Ensemble simulation and reports the energies of multiple states.

    Parameters
    ----------
        states : pandas.DataFrame_
            A DataFrame containing context global parameters (column names) and sets of values
            thereof. The potential energy will be reported for every state these parameters define.
            If one of the variables is named as `weight`, then its set of values will be assigned
            to every state as an importance sampling weight. Otherwise, all states will have
            identical weights. States which are supposed to only have their energies reported, with
            no actual visits, can have their weights set up to `-inf`.
        temperature : unit.Quantity
            The system temperature.

    Keyword Args
    ------------
        reportsPerExchange : int, optional, default=1
            The number of reports between attempts to exchange the global parameter state, that is,
            the exchange interval measured in units of report intervals.

    """
    def __init__(self, file, reportInterval, states, temperature, **kwargs):
        self._parameter_states = states.copy()
        self._nstates = len(states.index)
        self._reports_per_exchange = kwargs.pop('reportsPerExchange', 1)
        super().__init__(file, reportInterval, **kwargs)
        if 'weight' in states:
            self._weights = self._parameter_states.pop('weight').values
            finite = np.where(np.isfinite(self._weights))[0]
            self._first_state = finite[0]
            self._last_state = finite[-1]
        else:
            self._weights = np.zeros(self._nstates)
            self._first_state = 0
            self._last_state = self._nstates - 1
        kT = (unit.MOLAR_GAS_CONSTANT_R*temperature).value_in_unit(unit.kilojoules_per_mole)
        self._beta = 1.0/kT
        self._nreports = 0
        self._overall_visits = np.zeros(self._nstates, dtype=int)
        self._downhill_visits = np.zeros(self._nstates, dtype=int)
        self._probability_accumulators = np.zeros(self._nstates)
        self._downhill = False
        self._counting_started = False
        self._regime_change = []

    def _initialize(self, simulation, state):
        headers = ['step', 'state']
        for index in self._parameter_states.index:
            headers.append('Energy[{}] (kJ/mole)'.format(index))
        print(*headers, sep=self._separator, file=self._out)

    def _register_visit(self, state):
        if self._downhill:
            if state == self._first_state:
                self._downhill = False
                self._regime_change.append(self._nreports)
        elif state == self._last_state:
            self._downhill = True
            self._regime_change.append(self._nreports)
        if self._counting_started:
            self._overall_visits[state] += 1
            if self._downhill:
                self._downhill_visits[state] += 1
        else:
            self._counting_started = self._downhill is True

    def _generateReport(self, simulation, state):
        energies = np.zeros(self._nstates)
        original = dict()
        for name in self._parameter_states.columns:
            original[name] = simulation.context.getParameter(name)
        latest = original.copy()
        for i, (index, row) in enumerate(self._parameter_states.iterrows()):
            for name, value in row.items():
                if value != latest[name]:
                    simulation.context.setParameter(name, value)
                    latest[name] = value
            energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
            energies[i] = energy.value_in_unit(unit.kilojoules_per_mole)
        self._nreports += 1
        exponents = self._weights - self._beta*energies
        probabilities = np.exp(exponents - np.amax(exponents))
        probabilities /= np.sum(probabilities)
        self._probability_accumulators += probabilities
        if self._nreports % self._reports_per_exchange == 0:
            state = np.random.choice(self._nstates, p=probabilities)
            for name, value in self._parameter_states.iloc[state].items():
                if value != latest[name]:
                    simulation.context.setParameter(name, value)
            self._register_visit(state)
        print(simulation.currentStep, state, *energies, sep=self._separator, file=self._out)

    def _isochronal_delta(self, f, n):
        N = len(f)
        b = 3/(n*(n+1)*(2*n+1))
        seq = np.arange(1, n+1)
        a = (b/2)*np.array([n*(n+1)-k*(k-1) for k in seq])
        ind = np.argsort(f)
        fa = f[ind]
        delta = np.empty(N)
        delta[0] = -fa[0]/2 + np.sum(a*fa[1:n+1])
        for i in range(1, N-1):
            delta[i] = b*np.sum([k*(fa[min(i+k, N-1)] - fa[max(i-k, 0)]) for k in seq])
        delta[N-1] = fa[N-1]/2 - np.sum(np.flip(a)*fa[N-n-1:N-1])
        delta[ind] = delta
        return delta

    def read_csv(self, file, **kwargs):
        comment = kwargs.pop('comment', '#')
        separator = kwargs.pop('sep', self._separator)
        df = pd.read_csv(file, comment=comment, sep=separator, **kwargs)
        energies = np.zeros(self._nstates)
        for index, row in df.iterrows():
            state = int(row['state'])
            for i in self._parameter_states.index:
                energies[i] = row['Energy[{}] (kJ/mole)'.format(i)]
            self._nreports += 1
            exponents = self._weights - self._beta*energies
            probabilities = np.exp(exponents - np.amax(exponents))
            probabilities /= np.sum(probabilities)
            self._probability_accumulators += probabilities
            if self._nreports % self._reports_per_exchange == 0:
                self._register_visit(state)

    def state_sampling_analysis(self, staging_variable=None, to_file=True, isochronal_n=2):
        """
        Build histograms of states visited during the overall process as well as during downhill
        walks.

        Returns
        -------
            pandas.DataFrame_

        """
        mask = self._overall_visits > 0
        frame = pd.DataFrame(self._parameter_states)[mask]
        histogram = self._overall_visits[mask]
        downhill_fraction = self._downhill_visits[mask]/histogram
        weight = self._weights[mask]
        frame['weight'] = weight
        frame['histogram'] = histogram/np.sum(histogram)
        frame['downhill_fraction'] = downhill_fraction
        if self._counting_started:
            probability = self._probability_accumulators[mask]/self._nreports
            free_energy = weight - np.log(probability)
            free_energy -= free_energy[0]
            delta = self._isochronal_delta(downhill_fraction, isochronal_n)
            isochronal_weight = weight + 0.5*np.log(delta/probability)
            frame['free_energy'] = free_energy
            frame['isochronal_histogram'] = np.sqrt(delta*probability)
            frame['isochronal_weight'] = isochronal_weight - isochronal_weight[0]
            if staging_variable is not None:
                x = frame[staging_variable].values
                f = downhill_fraction
                n = len(x)
                optimal_pdf = np.sqrt(np.diff(f)/np.diff(x))      # Stepwise optimal PDF
                area = optimal_pdf*np.diff(x)                     # Integral in each interval
                optimal_cdf = np.cumsum(area)/np.sum(area)        # Piecewise linear optimal CDF
                optimal_x = np.interp(np.linspace(0, 1, n), np.insert(optimal_cdf, 0, 0), x)
                frame['staging_{}'.format(staging_variable)] = optimal_x
                frame['staging_weight'] = np.interp(optimal_x, x, free_energy)
        if to_file:
            print('# {0} State Sampling Analysis {0}'.format('-'*40), file=self._out)
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print('# ' + frame.to_string(index=False).replace('\n', '\n# '), file=self._out)
        return frame

    def walking_time_analysis(self, history=False, to_file=True):
        times = np.diff(np.array(self._regime_change))
        downhill = self._reportInterval*times[0::2]
        uphill = self._reportInterval*times[1::2]
        if history:
            df = pd.DataFrame({'downhill': pd.Series(downhill),
                               'uphill': pd.Series(uphill)})
            print('# {0} Walking Time History {0}'.format('-'*10), file=self._out)
            print('# ' + df.to_string().replace('\n', '\n# '), file=self._out)
        df = pd.DataFrame(index=['count', 'mean time'],
                          columns=['downhill', 'uphill'],
                          data=[[downhill.size, uphill.size],
                                [downhill.mean(), uphill.mean()]],
                          dtype='object')
        if to_file:
            print('# {0} Walking Time Analysis {0}'.format('-'*10), file=self._out)
            print('# ' + df.to_string().replace('\n', '\n# '), file=self._out)
        return df
