"""
.. module:: forces
   :platform: Unix, Windows
   :synopsis: a module for defining integrator classes.

.. moduleauthor:: Charlles R. A. Abreu <abreu@eq.ufrj.br>

"""

from simtk import openmm


class VelocityVerletIntegrator(openmm.CustomIntegrator):
    """
    A simple Verlocity Verlet integrator, with coordinates and momenta evaluated synchronously.

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
