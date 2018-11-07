"""
.. module:: reporters
   :platform: Unix, Windows
   :synopsis: a module for defining OpenMM reporter classes.

.. moduleauthor:: Charlles R. A. Abreu <abreu@eq.ufrj.br>

.. _pandas.DataFrame: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html

"""

import numpy as np
import pandas as pd
from simtk import unit

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


class Reporter(object):
    """
    Base class for reporters.

    """
    def __init__(self, file, interval, separator=","):
        self._interval = interval
        self._separator = separator
        self._fileWasOpened = isinstance(file, str)
        if self._fileWasOpened:
            # Detect the desired compression scheme from the filename extension
            # and open all files unbuffered
            if file.endswith(".gz"):
                if not have_gzip:
                    raise RuntimeError("Cannot write .gz file because Python could not import gzip library")
                self._out = gzip.GzipFile(fileobj=open(file, "wb", 0))
            elif file.endswith(".bz2"):
                if not have_bz2:
                    raise RuntimeError("Cannot write .bz2 file because Python could not import bz2 library")
                self._out = bz2.BZ2File(file, "w", 0)
            else:
                self._out = open(file, "w")
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
        print(self._separator.join("{}".format(v) for v in values), file=self._out)
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


class MultistateEnergyReporter(Reporter):
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
    states : dict(string: list(number)) or _pandas.DataFrame
        The names (keys) and set of values of global variables which define the thermodynamic
        states. All provided value lists must have the same size.
    separator : str, optional, default=","
        By default the data is written in comma-separated-value (CSV) format, but you can specify a
        different separator to use.
    describeStates : bool, optional, default=False
        If `True`, the first lines of the output file will contain lines describing the names and
        values of the state-defining variables at each state.

    """
    def __init__(self, file, interval, states, separator=",", describeStates=False):
        super().__init__(file, interval, separator)
        self._states = states if isinstance(states, pd.DataFrame) else pd.DataFrame.from_dict(states)
        self._nstates = len(self._states.index)
        self._variables = self._states.columns.values
        self._describe = describeStates

    def _headers(self):
        return ["step"]+["E{}".format(index) for (index, value) in self._states.iterrows()]

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
                print("# State {}:".format(index),
                      ", ".join("{}={}".format(v, value[v]) for v in self._variables),
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
    states : dict(string: list(number)) or _pandas.DataFrame
        The names (keys) and set of values of global variables which define the thermodynamic
        states. All provided value lists must have the same size.
    weights : list(number)
        The importance weights for biasing the probability of picking each state in an exchange.
    temperature : unit.Quantity
        The system temperature.
    separator : str, optional, default=","
        By default the data is written in comma-separated-value (CSV) format, but you can specify a
        different separator to use.
    describeStates : bool, optional, default=False
        If `True`, the first lines of the output file will contain lines describing the names and
        values of the state-defining variables at each state.

    """
    def __init__(self, file, exchangeInterval, reportInterval, states, weights,
                 temperature, separator=",", describeStates=False):
        super().__init__(file, exchangeInterval, states, separator, describeStates)
        self._reportInterval = reportInterval
        self._weights = np.array(weights)
        kT = (unit.BOLTZMANN_CONSTANT_kB*unit.AVOGADRO_CONSTANT_NA*temperature).value_in_unit(unit.kilojoules_per_mole)
        self._beta = 1/kT
        self._currentState = -1
        self._exchangeCount = 0

    def _headers(self):
        return ["step", "state"]+["E{}".format(index) for (index, value) in self._states.iterrows()]

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
