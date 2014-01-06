# TRAINFILE / TESTFILE are the names of the official train and test datasets
TRAINFILE = 'data/train.csv'
TESTFILE = 'data/test.csv'

# GEODATA is the name of the csv file containing mined Geodata indexed by issue ID
GEODATA = 'data/geodata.csv'

# USE_BRYANS_DATA is a boolean, if True then use Bryan's updated dataset
# when generating the model and must also include BRYAN_TRAIN and BRYAN_TEST 
# locations for Bryan's train and test files
USE_BRYANS_DATA = True
BRYAN_TRAIN = "data/train_addr_inc_pop.csv"
BRYAN_TEST = "data/test_addr_inc_pop.csv"

# CACHEDIR is the loaction of the cache
CACHEDIR = 'cache/'

# CACHETYPE is the type of caching to use - either pickle or joblib
CACHETYPE = 'joblib'

# SUBMITDIR is the location where submissions will be stored
SUBMITDIR = 'submissions/'

# SETTINGS is the name of the model and dataset settings json
SETTINGS = 'settings.json'

## Helper methods ##
import json

def json_configs(type, name):
    """
    Base method that extracts the configuration info from the json file defined
    in SETTINGS
    
    Args:
        type - the name of the type of configuration object to look in
        name - the name of the object whose configs will be extracted
    
    Returns:
        a dict containing the settings for the object of type and name
    
    Raises:
        a value error if type or name are not defined in the SETTINGS json file
    """
    f = open(SETTINGS)
    configs = json.load(f)[type]
    f.close()
    
    if name not in configs:
        raise ValueError('Unable to find configuration for %s %s' % (type, name))
    
    return configs[name]
    
def model_configs(name):
    """
    Wrapper for json_configs where type is 'models'
    """
    return json_configs('models', name)
    
def dataset_configs(name):
    """
    Wrapper for json_configs where type is 'datasets'
    """
    return json_configs('datasets', name)
