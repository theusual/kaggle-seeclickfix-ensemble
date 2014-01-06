"""
Functions for data IO
"""
__author__ = 'Bryan Gregory'
__email__ = 'bryan.gregory1@gmail.com'
__date__ = '09-06-2013'

#Internal modules
import utils
#Start logger to record all info, warnings, and errors to Logs/logfile.log
log = utils.start_logging(__name__)

#External modules
import json
import csv
import gc
import pandas as pd
import time
import os
from datetime import datetime
from sklearn.externals import joblib

#import JSON data into a dict
def load_json(file_path):
    return [json.loads(line) for line in open(file_path)]

#import delimited flat file into a list
def load_flatfile(file_path, delimiter=''):
    temp_array = []
    #if no delimiter is specified, try to use the built-in delimiter detection
    if delimiter == '':
        csv_reader = csv.reader(open(file_path))
    else:
        csv_reader = csv.reader(open(file_path),delimiter)
    for line in csv_reader:
        temp_array += line
    return temp_array #[line for line in csv_reader]

#import delimited flat file into a pandas dataframe
def load_flatfile_to_df(file_path, delimiter=''):
    #if no delimiter is specified, try to use the built-in delimiter detection
    if delimiter == '':
        return pd.read_csv(file_path)
    else:
        return pd.read_csv(file_path, delimiter)

def save_predictions(df,target,model_name='',directory='Submits/',estimator_class='',note=''):
    timestamp = datetime.now().strftime('%m-%d-%y_%H%M')
    filename = directory+timestamp+'--'+model_name+'_'+estimator_class+'_'+note+'.csv'
    #---Perform any manual predictions cleanup that may be necessary---#

    #Save predictions
    try:
        df[target] = [x[0] for x in df[target]]
    except IndexError:
        df[target] = [x for x in df[target]]
    df.ix[:,['id',target]].to_csv(filename, index=False)
    log.info('Submission file saved: %s' % filename)

def save_combined_predictions(df,directory,filename,note=''):
    #If previous combined predictions already exist, archive existing ones by renaming to append datetime
    try:
        modified_date = time.strptime(time.ctime(os.path.getmtime(directory+filename)), '%a %b %d %H:%M:%S %Y')
        modified_date = datetime.fromtimestamp(time.mktime(modified_date)).strftime('%m-%d-%y_%H%M')
        archived_file = directory+'Archive/'+filename[:len(filename)-4]+'--'+modified_date+'.csv'
        os.rename(directory+filename,archived_file)
        log.info('File already exists with given filename, archiving old file to: '+ archived_file)
    except WindowsError:
        pass
    #Save predictions
    df.to_csv(directory+filename, index=False)
    log.info('Predictions saved: %s' % filename)

def save_cached_object(object, filename, directory='Cache/'):
    """Save cached objects in pickel format using joblib compression.
       If a previous cached file exists, then get its modified date and append it to filename and archive it
    """
    if filename[-4:] != '.pkl':
        filename = filename+'.pkl'
    try:
        modified_date = time.strptime(time.ctime(os.path.getmtime(directory+filename)), '%a %b %d %H:%M:%S %Y')
        modified_date = datetime.fromtimestamp(time.mktime(modified_date)).strftime('%m-%d-%y_%H%M')
        archived_file = directory+'Archive/'+filename[:len(filename)-4]+'--'+modified_date+'.pkl'
        os.rename(directory+filename,archived_file)
        log.info('Cached object already exists with given filename, archiving old object to: '+ archived_file)
    except WindowsError:
        pass
    joblib.dump(object, directory+filename, compress=9)
    log.info('New object cached to: '+directory+filename)

def load_cached_object(filename, directory='Cache/'):
    if filename[-4:] != '.pkl':
        filename = filename+'.pkl'
    try:
        object = joblib.load(directory+filename)
        log.info('Successfully loaded object from: '+directory+filename)
    except IOError:
        log.info('Cached object does not exist: '+directory+filename)
    return object

def save_text_features(output_file, feature_names):
	o_f = open( output_file, 'wb' )
	feature_names = '\n'.join( feature_names )
	o_f.write( feature_names )