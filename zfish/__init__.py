"""Top-level package for zfish."""

__author__ = """Max Timo Hess"""
__email__ = 'max.hess@mls.uzh.ch'
__version__ = '0.1.0'

import logging

logger = logging.getLogger(__name__)

# if we don't have any handlers, set one up
if not logger.handlers:
    # configure stream handler
    log_fmt = logging.Formatter(
        "[%(levelname)s] %(message)s",
        # "[%(levelname)s][%(asctime)s] %(message)s",
        # datefmt="%Y/%m/%d %I:%M:%S",
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_fmt)

    logger.addHandler(console_handler)
    logger.setLevel(logging.DEBUG)