"""
.. module:: forces
   :platform: Unix, Windows
   :synopsis: a module for defining the class :class:`force`.

.. moduleauthor:: Charlles R. A. Abreu <abreu@eq.ufrj.br>

"""

from atomsmm.utils import InputError


class force:
    """
    A sample of configurations distributed according to a :term:`PDF`
    proportional to ``exp(-u(x))``. Each configuration ``x`` is represented
    by a set of collective variables from which one can evaluate the reduced
    potential ``u(x)``, as well as other properties of interest.

    Parameters
    ----------
        dataset : pandas.DataFrame
            A data frame whose column names are collective variables used to
            represent the sampled comfigurations. The rows must contain a time
            series of these variables, obtained by simulating the system at a
            state with known reduced potential.
        potential : str
            A mathematical expression defining the reduced potential of the
            simulated state. This must be a function of the column names in
            `dataset` and can also depend on external parameters passed as
            keyword arguments (see below).
        acfun : str, optional, default=potential
            A mathematical expression defining a property to be used for
            :term:`OBM` autocorrelation analysis and effective sample size
            calculation. It must be a function of the column names in `dataset`
            and can also depend on external parameters passed as keyword
            arguments (see below).
        batchsize : int, optional, default=sqrt(len(dataset))
            The size of each batch (window) to be used in the :term:`OBM`
            analysis. If omitted, then the batch size will be the integer
            part of the square root of the sample size.
        **constants : keyword arguments
            A set of keyword arguments passed as name=value, aimed to define
            external parameter values for the evaluation of the mathematical
            expressions in `potential` and `acfun`.

    """

    def __init__(self, dataset, potential, acfun=None, batchsize=None, **constants):
        pass
