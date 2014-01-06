"""
Generic utility functions useful in data-mining projects
"""
__author__ = 'Bryan Gregory'
__email__ = 'bryan.gregory1@gmail.com'
__date__ = '09-06-2013'

import logging
import json

def start_logging(scope_name):
    #Set logger to display INFO level and above
    log = logging.getLogger(scope_name)
    log.setLevel(logging.INFO)

    file_log_handler = logging.FileHandler('Logs/logfile.log')
    log.addHandler(file_log_handler)
    stderr_log_handler = logging.StreamHandler()
    log.addHandler(stderr_log_handler)

    # Append date/time
    formatter = logging.Formatter('%(asctime)s--%(levelname)s--%(module)s--%(message)s')
    file_log_handler.setFormatter(formatter)
    stderr_log_handler.setFormatter(formatter)
    return log

def line_break():
    """Returns a standard line break for improved print/log readability
    """
    line_break='============================================'
    return line_break

def load_settings(filename='SETTINGS.json'):
    """Load program settings and model settings to dict variables
    """
    log = start_logging(__name__)
    log.info('=============LOADING SETTINGS===============')
    settings = json.loads(open(filename).read())
    log.info('ENVIRONMENT SETTINGS: ')
    log.info(settings)
    return settings