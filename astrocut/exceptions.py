# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
Custom exceptions used in the astrocut classes
"""

from astropy.utils.exceptions import AstropyWarning


class InvalidQueryError(Exception):
    """
    Errors related to invalid queries.
    """
pass


class InputWarning(AstropyWarning):
    """
    Warning to be issued when use input is incorrect in
    some way but doesn't prevent the function from running.
    """
pass

class TypeWarning(AstropyWarning):
    """
    Warnings to do with data types.
    """
pass
