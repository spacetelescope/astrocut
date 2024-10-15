# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""This module configures the astrocut logger to use the astropy logging system."""

import logging

from astropy.logger import AstropyLogger


def setup_logger():
    log = logging.getLogger()
    orig_logger_cls = logging.getLoggerClass()
    logging.setLoggerClass(AstropyLogger)
    try:
        log = logging.getLogger('astrocut')
        log._set_defaults()
    finally:
        logging.setLoggerClass(orig_logger_cls)
    return log
