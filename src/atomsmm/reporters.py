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

from .systems import ComputingSystem
from .utils import InputError

try:
    import bz2
    have_bz2 = True
except ModuleNotFoundError:
    have_bz2 = False

try:
    import gzip
    have_gzip = True
except ModuleNotFoundError:
    have_gzip = False


class _AtomsMM_Reporter(openmm.app.StateDataReporter):
    """
    Base class for reporters.

    Keyword Args
    ------------
        extra : str or file, optional, default=None
            Extra file to write to, specified as a file name or a file object.

    """
    def __init__(self, file, reportInterval, **kwargs):
        extra = kwargs.pop('extra', None)
        if extra is None:
            super().__init__(file, reportInterval, **kwargs)
        else:
            super().__init__(self._MultiStream([file, extra]), reportInterval, **kwargs)

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


class ExtendedStateDataReporter(_AtomsMM_Reporter):
    """
    An extension of OpenMM's StateDataReporter_ class, which outputs information about a simulation,
    such as energy and temperature, to a file.

    All original functionalities of StateDataReporter_ are preserved, while the following ones are
    added:

    1. Report the Coulomb contribution of the potential energy:

        The Coulomb contribution includes both real- and reciprocal-space terms.

    2. Report the internal virial of a fully-flexible system:

        .. math::
            W = -\\sum_{i,j} r_{ij} E^\\prime(r_{ij}),

        where :math:`E^\\prime(r)` is the derivative of the pairwise interaction potential as a
        function of the distance between to atoms. This includes van der Waals, Coulomb, and
        bond-stretching interactions.

    3. Report the internal pressure of a fully-flexible system:

        .. math::
            P = \\frac{2K + W}{3 V},

        where :math:`K` is the total kinetic energy, :math:`W` is the internal virial, and :math:`V`
        is the volume of the system.

    4. Allow specification of an extra file for reporting.

        This can be used for replicating a report simultaneously to `sys.stdout` and to a file
        using a unique reporter.

    Keyword Args
    ------------
        coulombEnergy : bool, optional, default=False
            Whether to write the Coulomb contribution of the potential energy to the file.
        virial : bool, optional, default=False
            Whether to write the total internal virial to the file.
        pressure : bool, optional, default=False
            Whether to write the internal pressure to the file.
        computer : :class:`~atomsmm.systems.ComputingSystem`, optional, default=None
            A system designed to compute the internal virial. This is mandatory if keyword `virial`
            or `pressure` is set to `True`.
        extra : str or file, optional, default=None
            Extra file to write to, specified as a file name or a file object.

    """
    def __init__(self, file, reportInterval, **kwargs):
        self._coulombEnergy = kwargs.pop('coulombEnergy', False)
        self._virial = kwargs.pop('virial', False)
        self._pressure = kwargs.pop('pressure', False)
        self._computer = kwargs.pop('computer', None)
        super().__init__(file, reportInterval, **kwargs)
        if self._coulombEnergy or self._virial or self._pressure:
            if not isinstance(self._computer, ComputingSystem):
                raise InputError('ComputingSystem is required')
            if (self._virial or self._pressure) and not self._computer._okVirial:
                raise RuntimeError('cannot compute virial/pressure for system with constraints')
            self._requiresInitialization = True
            self._backSteps = -sum([self._speed, self._elapsedTime, self._remainingTime])
            self._needsPositions = True

    def _constructHeaders(self):
        headers = super()._constructHeaders()
        if self._coulombEnergy:
            headers.insert(self._backSteps, 'Coulomb Energy (kJ/mole)')
        if self._virial:
            headers.insert(self._backSteps, 'Virial (kJ/mole)')
        if self._pressure:
            headers.insert(self._backSteps, 'Pressure (atm)')
        return headers

    def _constructReportValues(self, simulation, state):
        values = super()._constructReportValues(simulation, state)
        if self._coulombEnergy or self._virial or self._pressure:
            if self._requiresInitialization:
                integrator = openmm.CustomIntegrator(0)
                platform = simulation.context.getPlatform()
                properties = dict()
                for name in platform.getPropertyNames():
                    properties[name] = platform.getPropertyValue(simulation.context, name)
                self._computeContext = openmm.Context(self._computer, integrator, platform, properties)
                self._requiresInitialization = False
            context = self._computeContext
            context.setPositions(state.getPositions())
            if self._coulombEnergy:
                energy = context.getState(getEnergy=True, groups=self._computer._coulombSet).getPotentialEnergy()
                values.insert(self._backSteps, energy.value_in_unit(unit.kilojoules_per_mole))
            if self._virial or self._pressure:
                box = state.getPeriodicBoxVectors()
                context.setPeriodicBoxVectors(*box)
                virial = context.getState(getEnergy=True, groups=self._computer._virialSet).getPotentialEnergy()
                if self._virial:
                    values.insert(self._backSteps, virial.value_in_unit(unit.kilojoules_per_mole))
                if self._pressure:
                    dNkT = 2*state.getKineticEnergy()
                    volume = box[0][0]*box[1][1]*box[2][2]
                    pressure = (dNkT + virial)/(3*volume*unit.AVOGADRO_CONSTANT_NA)
                    values.insert(self._backSteps, pressure.value_in_unit(unit.atmospheres))
        return values


class AtomsMMReporter(object):
    """
    Base class for reporters.

    """
    def __init__(self, file, interval, separator=','):
        self._interval = interval
        self._separator = separator
        self._fileWasOpened = isinstance(file, str)
        if self._fileWasOpened:
            # Detect the desired compression scheme from the filename extension
            # and open all files unbuffered
            if file.endswith('.gz'):
                if not have_gzip:
                    raise RuntimeError('Cannot write .gz file because Python could not import gzip library')
                self._out = gzip.GzipFile(fileobj=open(file, 'wb', 0))
            elif file.endswith('.bz2'):
                if not have_bz2:
                    raise RuntimeError('Cannot write .bz2 file because Python could not import bz2 library')
                self._out = bz2.BZ2File(file, 'w', 0)
            else:
                self._out = open(file, 'w')
        else:
            self._out = file
        self._requiresInitialization = True
        self._needsPositions = False
        self._needsVelocities = False
        self._needsForces = False
        self._needEnergy = False

    def __del__(self):
        if self._fileWasOpened:
            self._out.close()

    def _flush(self, values):
        print(self._separator.join('{}'.format(v) for v in values), file=self._out)
        try:
            self._out.flush()
        except AttributeError:
            pass

    def _initialize(self, simulation):
        self._requiresInitialization = False

    def describeNextReport(self, simulation):
        """
        Gets information about the next report this object will generate.

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
        if self._requiresInitialization:
            self._initialize(simulation)

        steps = self._interval - simulation.currentStep % self._interval
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
        pass


class MultistateEnergyReporter(AtomsMMReporter):
    """
    Reports the system energy at multiple thermodynamic states, writing the results to a file. To
    use it, create a MultistateEnergyReporter, then add it to the Simulation's list of reporters.

    Parameters
    ----------
    file : string or file
        The file to write to, specified as a file name or file object. One can try to use extensions
        `.gz` or `.bz2` in order to generate a compressed file, depending on the availability of the
        required Python packages.
    interval : int
        The interval (in time steps) at which to write reports.
    states : dict(string: list(number)) or pandas.DataFrame_
        The names (keys) and set of values of global variables which define the thermodynamic
        states. All provided value lists must have the same size.
    separator : str, optional, default=','
        By default the data is written in comma-separated-value (CSV) format, but you can specify a
        different separator to use.
    describeStates : bool, optional, default=False
        If `True`, the first lines of the output file will contain lines describing the names and
        values of the state-defining variables at each state.

    """
    def __init__(self, file, interval, states, separator=',', describeStates=False):
        super().__init__(file, interval, separator)
        self._states = states if isinstance(states, pd.DataFrame) else pd.DataFrame.from_dict(states)
        self._nstates = len(self._states.index)
        self._variables = self._states.columns.values
        self._describe = describeStates

    def _headers(self):
        return ['step']+['E{}'.format(index) for (index, value) in self._states.iterrows()]

    def _initialize(self, simulation):
        stateVariables = set(self._variables)
        self._dependentForces = list()
        self._forceGroups = list()
        lastForceGroup = 0
        for (index, force) in enumerate(simulation.system.getForces()):
            group = force.getForceGroup()
            lastForceGroup = max(lastForceGroup, group)
            try:
                n = force.getNumGlobalParameters()
                dependencies = set(force.getGlobalParameterName(i) for i in range(n))
                if dependencies & stateVariables:
                    self._dependentForces.append(force)
                    self._forceGroups.append(group)
            except AttributeError:
                pass
        self._availableForceGroup = lastForceGroup + 1
        if self._describe:
            for (index, value) in self._states.iterrows():
                print('# State {}:'.format(index),
                      ', '.join('{}={}'.format(v, value[v]) for v in self._variables),
                      file=self._out)
        self._flush(self._headers())
        self.report(simulation, None)
        self._requiresInitialization = False

    def _multistateEnergies(self, context):
        energies = list()
        originalValues = [context.getParameter(variable) for variable in self._variables]
        for force in self._dependentForces:
            force.setForceGroup(self._availableForceGroup)
        for (index, value) in self._states.iterrows():
            for variable in self._variables:
                context.setParameter(variable, value[variable])
            state = context.getState(getEnergy=True, groups=set([self._availableForceGroup]))
            energies.append(state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole))
        for (force, group) in zip(self._dependentForces, self._forceGroups):
            force.setForceGroup(group)
        for (variable, value) in zip(self._variables, originalValues):
            context.setParameter(variable, value)
        return energies

    def report(self, simulation, state):
        self._flush([simulation.currentStep] + self._multistateEnergies(simulation.context))


class ExpandedEnsembleReporter(MultistateEnergyReporter):
    """
    Performs an expanded ensemble simulation by walking through multiple thermodynamic states and
    writes the system energy at these states to a file. To use it, create a ExpandedEnsembleReporter
    and it to the Simulation's list of reporters.

    Parameters
    ----------
    file : string or file
        The file to write to, specified as a file name or file object. One can try to use extensions
        `.gz` or `.bz2` in order to generate a compressed file, depending on the availability of the
        required Python packages.
    exchangeInterval : int
        The interval (in units of time steps) at which to try changes in thermodynamic states.
    reportInterval : int
        The interval (in units of exchange intervals) at which to write reports.
    states : dict(string: list(number)) or pandas.DataFrame_
        The names (keys) and set of values of global variables which define the thermodynamic
        states. All provided value lists must have the same size.
    weights : list(number)
        The importance weights for biasing the probability of picking each state in an exchange.
    temperature : unit.Quantity
        The system temperature.
    separator : str, optional, default=','
        By default the data is written in comma-separated-value (CSV) format, but you can specify a
        different separator to use.
    describeStates : bool, optional, default=False
        If `True`, the first lines of the output file will contain lines describing the names and
        values of the state-defining variables at each state.

    """
    def __init__(self, file, exchangeInterval, reportInterval, states, weights,
                 temperature, separator=',', describeStates=False):
        super().__init__(file, exchangeInterval, states, separator, describeStates)
        self._reportInterval = reportInterval
        self._weights = np.array(weights)
        kT = (unit.BOLTZMANN_CONSTANT_kB*unit.AVOGADRO_CONSTANT_NA*temperature).value_in_unit(unit.kilojoules_per_mole)
        self._beta = 1/kT
        self._currentState = -1
        self._exchangeCount = 0

    def _headers(self):
        return ['step', 'state']+['E{}'.format(index) for (index, value) in self._states.iterrows()]

    def report(self, simulation, state):
        energies = np.array(self._multistateEnergies(simulation.context))
        probabilities = np.exp(-self._beta*energies + self._weights)
        probabilities /= sum(probabilities)
        newState = np.random.choice(range(self._nstates), p=probabilities)
        if newState != self._currentState:
            for (variable, value) in self._states.iloc[newState].to_dict().items():
                simulation.context.setParameter(variable, value)
            self._currentState = newState
        if self._exchangeCount % self._reportInterval == 0:
            self._flush([simulation.currentStep, self._currentState] + energies.tolist())
        self._exchangeCount += 1
