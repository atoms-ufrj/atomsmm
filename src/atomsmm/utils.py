"""
.. module:: utils
   :platform: Unix, Windows
   :synopsis: a module for auxiliary tasks.

.. moduleauthor:: Charlles R. A. Abreu <abreu@eq.ufrj.br>

"""

class InputError(Exception):
    def __init__(self, msg):
        super(InputError, self).__init__("\033[1;31m" + msg + "\033[0m")
