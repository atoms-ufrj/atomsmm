"""
.. module:: forces
   :platform: Unix, Windows
   :synopsis: a module for defining integrator classes.

.. moduleauthor:: Charlles R. A. Abreu <abreu@eq.ufrj.br>

"""

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
        timestep : Number or unit.Quantity
           The step size with which to integrate the system (in picoseconds or in an explicitly
           specified time unit).

    """
    def __init__(self, timestep):
        super(VelocityVerletIntegrator, self).__init__(timestep)
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

    """
    def __init__(self, temperature=298*unit.kelvin, time_constant=100/unit.picoseconds,
                 timestep=0.001*unit.picoseconds):
        pass
