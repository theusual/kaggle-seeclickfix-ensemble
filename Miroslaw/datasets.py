import pandas as pd
import numpy as np
import logging

import config, utils, data_transforms

logger = logging.getLogger(__name__)

def load_dataset(name):
    """
    Load dataset defined with name in SETTINGS json from cache. If not in cache
    create the dataset.
    
    Args:
        name - a string of a valid dataset defined in SETTINGS
    
    Returns:
        the dataset
    """
    try:
        dataset = utils.load_from_cache(name)
        logger.info('Loaded dataset %s from cache' % name)
    except IOError:
        dataset = make_dataset(name)
        utils.save_to_cache(dataset, name)
    return dataset

def make_dataset(name):
    """
    Create the dataset defined with name in SETTINGS json. A dataset definition
    in SETTINGS takes the form:
        { "DatasetName": {
            "input_data": ["NamesOfAnyDatasetsRequiredAsInput", ...],
            "transforms": [[ "TrainsformName1", { "TransformArg1": Value1, 
                                                  "TransformArg2": Value2, ... }].
                            ... ]
          }
        }
    "DatasetName" is the name that will be used for the dataset throughout the model
    this name is used for accessing configuration information, and when passing the
    dataset as input into other dataset definitions or model definitions
    
    "input_data" is a list of other dataset names that are required as input for
    the creation of the dataset
    
    "transforms" is a list of lists. Each sub-list must have length 2 and contain
    a transform name, which is a valid name of a function defined in data_transforms.py
    and a dict of arguments required to pass into the transform. All transforms defined
    in data_transforms.py must have the structure:
        
        def transform_name(input_data, **args):
            ...
            
    The "transforms" list allows for chaining of transforms for creating complex
    datasets. The input for the first transform in the chain is always defined in 
    "input_data", the next transform in the chain takes the output of the previous
    transform as input_data. 
    """
    cfgs = config.dataset_configs(name)
    data = [load_dataset(ds) for ds in cfgs['input_data']]
    
    if len(data) == 1:
        data = data[0]
    
    logger.info('Creating dataset %s' % name)
    for tname, args in cfgs['transforms']:
        try:
            transform = getattr(data_transforms, tname)
        except AttributeError:
            raise AttributeError('Unable to find transform \
                                   %s in data_transforms.py' % tname)
        logger.info('Applying %s on %s' % (tname, name))
        data = transform(data, **args)
    return data
