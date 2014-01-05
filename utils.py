"""
Generic utility functions useful in data-mining projects
"""
__author__ = 'Bryan Gregory'
__email__ = 'bryan.gregory1@gmail.com'
__date__ = '09-06-2013'

import json

def info(msg):
    """Display and log informational messages
    """
    print msg
    log(msg)

def log(msg):
    """Log to log file
    """
    #TODO: Implement logging
    pass

def line_break():
    """Print and log a standard line break
    """
    line_break='============================================'
    print line_break
    log(line_break)

def load_settings(filename='SETTINGS.json'):
    """Load environment settings to dict variables
    """
    line_break()
    info('=============LOADING SETTINGS===============')
    settings = json.loads(open(filename).read())
    line_break()
    return settings