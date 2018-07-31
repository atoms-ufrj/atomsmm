"""
.. module:: forces
   :platform: Unix, Windows
   :synopsis: a module for defining integrator classes.

.. moduleauthor:: Charlles R. A. Abreu <abreu@eq.ufrj.br>

"""

import math

from simtk import openmm
from simtk import unit


class Algorithm:
    def __init__(self):
        self.globalVariables = dict()
        self.perDofVariables = dict()

    def addVariables(self, integrator):
        for (name, value) in self.globalVariables.items():
            integrator.addGlobalVariable(name, value)
        for (name, value) in self.perDofVariables.items():
            integrator.addPerDofVariable(name, value)


class VelocityVerlet(Algorithm):
    """
    This class implements a simple Verlocity Verlet integration algorithm, in which coordinates and
    momenta are evaluated synchronously.

    .. note::
        In the original OpenMM VerletIntegrator_ class, the implemented algorithm is a leap-frog
        version of the Verlet method.

    .. _VerletIntegrator: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.VerletIntegrator.html

    """
    def __init__(self):
        super(VelocityVerlet, self).__init__()
        self.perDofVariables["x1"] = 0

    def addSteps(self, integrator, fraction=1):
        integrator.addUpdateContextState()
        integrator.addComputePerDof("v", "v+0.5*%s*dt*f/m" % fraction)
        integrator.addComputePerDof("x", "x+%s*dt*v" % fraction)
        integrator.addComputePerDof("x1", "x")
        integrator.addConstrainPositions()
        integrator.addComputePerDof("v", "v+0.5*dta*f/m+(x-x1)/dta; dta=%s*dt" % fraction)
        integrator.addConstrainVelocities()


class BussiDonadioParrinello(Algorithm):
    """
    This class implements the Stochastic Velocity Rescaling algorithm of Bussi, Donadio, and
    Parrinello :cite:`Bussi_2007`.

    .. warning::
        This integrator requires non-zero initial velocities for the system particles.

    Parameters
    ----------
        temperature : Number or unit.Quantity
            The temperature of the heat bath (in Kelvin).
        timeConstant : Number or unit.Quantity
            The characteristic time constant of thermostat action (in picoseconds).
        degreesOfFreedom : int
            The number of degrees of freedom in the system, which can be retrieved via function
            :func:`~atomsmm.utils.countDegreesOfFreedom`.

    """
    def __init__(self, temperature, timeConstant, degreesOfFreedom):
        super(BussiDonadioParrinello, self).__init__()
        self.globalVariables["R"] = 0
        self.globalVariables["Z"] = 0
        self.globalVariables["ready"] = 0
        self.globalVariables["TwoKE"] = 0
        self.globalVariables["factor"] = 0
        self.tau = timeConstant.value_in_unit(unit.picoseconds)
        self.dof = degreesOfFreedom
        kB = unit.BOLTZMANN_CONSTANT_kB*unit.AVOGADRO_CONSTANT_NA
        self.kT = (kB*temperature).value_in_unit(unit.kilojoules_per_mole)

    def addSteps(self, integrator, fraction=1):
        shape = (self.dof - 2 + self.dof % 2)/2
        d = shape - 1/3
        c = 1/math.sqrt(9*d)
        integrator.addComputeGlobal("ready", "0")
        integrator.beginWhileBlock("ready = 0")
        integrator.addComputeGlobal("R", "gaussian")
        integrator.addComputeGlobal("Z", "1+%s*R" % c)
        integrator.beginWhileBlock("Z <= 0")
        integrator.addComputeGlobal("R", "gaussian")
        integrator.addComputeGlobal("Z", "1+%s*R" % c)
        integrator.endBlock()
        integrator.addComputeGlobal("Z", "Z^3")
        integrator.addComputeGlobal("ready", "step(1-0.0331*y*y-u)")
        integrator.beginIfBlock("ready = 0")
        integrator.addComputeGlobal("ready", "step(0.5*y+%s*(1-Z+log(Z))-log(u))" % d)
        integrator.endBlock()
        integrator.endBlock()
        integrator.addComputeSum("TwoKE", "m*v*v")
        integrator.addComputeGlobal("R", "gaussian")
        expression = "sqrt(A+C*B*(R^2+sumRs)+2*sqrt(C*B*A)*R);"
        expression += "C = %s/TwoKE;" % self.kT
        expression += "B = 1-A;"
        expression += "A = exp(-dt*%s);" % (fraction/self.tau)
        expression += "sumRs = 2*%s*Z;" % d if self.dof % 2 == 0 else "sumRs = 2*%s*Z;" % d
        integrator.addComputeGlobal("factor", expression)
        integrator.addComputePerDof("v", "factor*v")


class VelocityVerletIntegrator(openmm.CustomIntegrator):
    """
    This class implements a simple Verlocity Verlet integrator, with coordinates and momenta
    evaluated synchronously.

    ..note:
        The original OpenMM VerletIntegrator_ class implements a leap-frog version of the Verlet
        method.

    .. _VerletIntegrator: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.VerletIntegrator.html

    Parameters
    ----------
        stepSize : Number or unit.Quantity
           The step size with which to integrate the system (in picoseconds or in an explicitly
           specified time unit).

    """
    def __init__(self, stepSize):
        super(VelocityVerletIntegrator, self).__init__(stepSize)
        algorithm = VelocityVerlet()
        algorithm.addVariables(self)
        algorithm.addSteps(self)


class BussiDonadioParrinelloIntegrator(openmm.CustomIntegrator):
    """
    This class implements the Stochastic Velocity Rescaling algorithm of Bussi, Donadio, and
    Parrinello :cite:`Bussi_2007`

    .. warning::
        This integrator requires non-zero initial velocities for the system particles.

    Parameters
    ----------
        temperature : Number or unit.Quantity
            The temperature of the heat bath (in Kelvin).
        frictionCoeff : Number or unit.Quantity
            The friction coefficient which couples the system to the heat bath (in inverse
            picoseconds).
        stepSize : Number or unit.Quantity
            The step size with which to integrate the system (in picoseconds).
        degreesOfFreedom : int
            The number of degrees of freedom in the system, which can be retrieved via function
            :func:`~atomsmm.utils.countDegreesOfFreedom`.

    """
    def __init__(self, temperature, frictionCoeff, stepSize, degreesOfFreedom, randomSeed=None):
        super(BussiDonadioParrinelloIntegrator, self).__init__(stepSize)
        if randomSeed is not None:
            self.setRandomNumberSeed(randomSeed)
        nve = VelocityVerlet()
        thermostat = BussiDonadioParrinello(temperature, 1.0/frictionCoeff, degreesOfFreedom)
        nve.addVariables(self)
        thermostat.addVariables(self)

        thermostat.addSteps(self, 1/2)
        nve.addSteps(self)
        thermostat.addSteps(self, 1/2)
