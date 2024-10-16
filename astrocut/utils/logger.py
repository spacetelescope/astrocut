# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""This module configures the astrocut logger."""

import logging


def setup_logger():
    """Set up a logger for astrocut"""
    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)  # default logging level

    # Create a console handler with format
    sh = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s: %(message)s [%(module)s]')
    sh.setFormatter(formatter)
    log.addHandler(sh)

    return log
