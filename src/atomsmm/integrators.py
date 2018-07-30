"""
.. module:: forces
   :platform: Unix, Windows
   :synopsis: a module for defining integrator classes.

.. moduleauthor:: Charlles R. A. Abreu <abreu@eq.ufrj.br>

"""

import math

from simtk import openmm
from simtk import unit


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
        self.addPerDofVariable("x1", 0)
        self.addUpdateContextState()
        self.addComputePerDof("v", "v+0.5*dt*f/m")
        self.addComputePerDof("x", "x+dt*v")
        self.addComputePerDof("x1", "x")
        self.addConstrainPositions()
        self.addComputePerDof("v", "v+0.5*dt*f/m+(x-x1)/dt")
        self.addConstrainVelocities()


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
        kT = unit.BOLTZMANN_CONSTANT_kB*unit.AVOGADRO_CONSTANT_NA*temperature/unit.kilojoules_per_mole
        self._addVariableDeclarations()
        self._addRescaleVelocities(stepSize/2, 1.0/frictionCoeff, degreesOfFreedom, kT)
        self.addUpdateContextState()
        self.addComputePerDof("v", "v+0.5*dt*f/m")
        self.addComputePerDof("x", "x+dt*v")
        self.addComputePerDof("x1", "x")
        self.addConstrainPositions()
        self.addComputePerDof("v", "v+0.5*dt*f/m+(x-x1)/dt")
        self.addConstrainVelocities()
        self._addRescaleVelocities(stepSize/2, 1.0/frictionCoeff, degreesOfFreedom, kT)

    def _addVariableDeclarations(self):
        self.addGlobalVariable("R", 0)
        self.addGlobalVariable("Z", 0)
        self.addGlobalVariable("ready", 0)
        self.addGlobalVariable("TwoKE", 0)
        self.addGlobalVariable("factor", 0)
        self.addPerDofVariable("x1", 0)

    def _addRescaleVelocities(self, dt, tau, dof, kT):
        shape = (dof - 2 + dof % 2)/2
        d = shape - 1/3
        c = 1/math.sqrt(9*d)
        self.addComputeGlobal("ready", "0")
        self.beginWhileBlock("ready = 0")
        self.addComputeGlobal("R", "gaussian")
        self.addComputeGlobal("Z", "1+%s*R" % c)
        self.beginWhileBlock("Z <= 0")
        self.addComputeGlobal("R", "gaussian")
        self.addComputeGlobal("Z", "1+%s*R" % c)
        self.endBlock()
        self.addComputeGlobal("Z", "Z^3")
        clause = "select(c1, 1, c2);"
        clause += "c1 = step(1-0.0331*y*y-u);"
        clause += "c2 = step(0.5*y+%s*(1-Z+log(Z))-log(u));" % d
        clause += "u = uniform;"
        clause += "y = R^2"
        self.addComputeGlobal("ready", clause)
        self.endBlock()
        self.addComputeSum("TwoKE", "m*v*v")
        self.addComputeGlobal("R", "gaussian")
        factor = "sqrt(A+C*B*(R^2+sumRs)+2*sqrt(C*B*A)*R);"
        factor += "C = %s/TwoKE;" % kT
        factor += "B = 1-A;"
        factor += "A = %s;" % math.exp(-dt/tau)
        if dof % 2 == 0:
            factor += "sumRs = 2*%s*Z;" % d
        else:
            factor += "sumRs = 2*%s*Z+gaussian^2;" % d
        self.addComputeGlobal("factor", factor)
        self.addComputePerDof("v", "factor*v")
