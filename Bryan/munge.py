"""
Functions for data loading, cleaning, and merging data
"""
__author__ = 'Bryan Gregory'
__email__ = 'bryan.gregory1@gmail.com'
__date__ = '11-19-2013'

#Internal modules
import utils
#Start logger to record all info, warnings, and errors to Logs/logfile.log
log = utils.start_logging(__name__)

#External modules
import numpy as np
import json
import datetime

#Clean the data of inconsistencies, bad date fields, bad data types, nested columns, etc.
def clean(df):
    #----Convert created_time to date object ----#
    df['created_time'] = [datetime.datetime.strptime(x, '%m/%d/%Y %H:%M') for x in df.created_time]

    #-----Parse time from date----#
    df['created_date'] = [x.date() for x in df.created_time]
    df['created_time'] = [x.time() for x in df.created_time]
    #df['created_date'] = [datetime.datetime.strptime(x[:10], '%Y-%m-%d') for x in df['created_time_orig']]
    #df['created_time'] = [datetime.datetime.strptime(x[11:], '%H:%M:%S') for x in df['created_time_orig']]
    df['month'] = [str(x.month) for x in df['created_date']]

    #---Fill short or missing text data----#
    df['description']=['___missing___' if len(str(df['description'][idx])) < 4 else df['description'][idx] for idx in df.index]

    #---Fill missing categorical data with NA---#
    df['tag_type'] = df.tag_type.fillna('NA')
    df['source'] = df.source.fillna('NA')

    #---recode to ASCII, which ignores any special non-linguistic characters ---#
    df['summary'] = [x.decode('iso-8859-1') for x in df.summary]
    df['description'] = [x.decode('iso-8859-1') for x in df.description]

    #---Clean problem characters in any text features---#
    df['summary'] = [x.replace('\n',' ') for x in df.summary]
    df['description'] = [x.replace('\n',' ') for x in df.description]
    df['summary'] = [x.replace('\r',' ') for x in df.summary]
    df['description'] = [x.replace('\r',' ') for x in df.description]
    df['summary'] = [x.replace('\\',' ') for x in df.summary]
    df['description'] = [x.replace('\\',' ') for x in df.description]
    df['summary'] = [x.replace('?',' ') for x in df.summary]
    df['description'] = [x.replace('?',' ') for x in df.description]

    #---Lower case any text features---#
    for idx in df.index:
        df.summary[idx] = df.summary[idx].lower()
        df.description[idx] = df.description[idx].lower()
        df.tag_type[idx] = df.tag_type[idx].lower()

    #----Transform target variables to allow predictions in RMSE space----#
    if 'num_votes' in df.keys():
        for target in ['num_views','num_votes','num_comments']:
            df[target] = np.log(df[target] + 1)

    #----Convert data types-----#

    #----Remove any unnecessary columns----#

    #----Reduce training dataset by parsing off sections of irrelevant training records----#
    #Use only months: 11/2012, 12/2012, 01/2013, 02/2013, 03/2013, 04/2013
    if 'num_votes' in df.keys():
        df = df[df.created_date > datetime.date(2012, 11, 1)]
        #df = df[df.month != '4']
    return df

def temporal_split(dfTrn, temporal_cutoff):
    dfTest = dfTrn[dfTrn.created_date >= datetime.date(temporal_cutoff[0],temporal_cutoff[1],temporal_cutoff[2])]
    dfTrn = dfTrn[dfTrn.created_date < datetime.date(temporal_cutoff[0],temporal_cutoff[1],temporal_cutoff[2])]
    return dfTrn, dfTest

def list_split(dfTrn,dfCVlist):
    dfTest = dfTrn.merge(dfCVlist, on='id', how='inner')
    dfTrn = dfTrn.merge(dfCVlist, on='id', how='left')
    dfTrn = dfTrn[np.isnan(dfTrn['dummy'])]
    del dfTrn['dummy']; del dfTest['dummy']
    return dfTrn, dfTest

def segment_data(dfTrn,dfTest,segment):
    if segment == 'remote_api_created':
        dfTrn = dfTrn[dfTrn.source == segment]
        dfTest = dfTest[dfTest.source == segment]
    if segment in ['Oakland']:
        #For Oakland, clean remote_api issues off of the test set.  Leave them in training set
        #due to marginally better CV performance with remote_api issues left in training set
        dfTrn = dfTrn[dfTrn.city == segment]
        dfTest = dfTest[dfTest.city == segment]
        dfTest = dfTest[np.logical_and(dfTest.city == segment,dfTest.source != 'remote_api_created')]
    if segment in ['Chicago']:
        dfTrn = dfTrn[np.logical_and(dfTrn.city == segment,dfTrn.source != 'remote_api_created')]
        dfTest = dfTest[np.logical_and(dfTest.city == segment,dfTest.source != 'remote_api_created')]
    if segment in ['New Haven','Richmond']:
        dfTrn = dfTrn[dfTrn.city == segment]
        dfTest = dfTest[dfTest.city == segment]
    return dfTrn, dfTest
