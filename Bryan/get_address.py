'''
About:
Uses the nominatim service hosted at both MapQuest and OSM.org to reverse lookup address info using long/lat coordinates
from data file,then outputs the same data file with the new address info added in columns.

Currently pulls zip code, street name, neighborhood (if available), and city.  Other available options that can be easily
added are: country, country_code, state, and county.

Includes the option to throttle (slow) the URL requests down to prevent overloading a connection and/or the nominatim
servers.  Also includes the option to only read certain lines of a file for processing large files in sections.

Requirements:
Python >2.4, PANDAS

Usage:
$ python get_address.py <input_file> <output_file> <log_file> <email_address> [start_line] [end_line] [throttle_time] [delimiter]
'''
__author__ = 'Bryan Gregory'
__email__ = 'bryan.gregory1@gmail.com'
__date__ = '10-02-2013'

import urllib2
import logging
import json
import sys
import time
import pandas as pd

def main():
    #---Command line params---#
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    log_file = sys.argv[3]
    email_address = sys.argv[4]
    try:
        start_line = int(sys.argv[5])
    except IndexError:
        start_line = 0
    try:
        end_line = int(sys.argv[6])
    except IndexError:
        end_line = 200000
    try:
        throttle_time = float(sys.argv[7])
    except IndexError:
        throttle_time = .1
    try:
        delimiter = sys.argv[8]
    except IndexError:
        delimiter = ""

    #---Configure logging settings---#
    logging.basicConfig(filename=log_file,level=logging.DEBUG)

    #---Load input data into a dataframe---#
    if delimiter == '':
        df = pd.read_csv(input_file)
    else:
        df = pd.read_csv(input_file, delimiter)

    #---Initialize the 3 dataframe fields that we are interested in from the OSM address object---#
    df['zipcode'] = "UKNOWN"
    df['street'] = "UKNOWN"
    df['city'] = "UKNOWN"
    df['neighborhood'] = "UKNOWN"

    #---Iterate through each record in df, sending the long/lat to OSM and receiving the address in return---#
    counter = 0
    source = 'MQ'
    for idx in df[start_line:end_line].index:
        counter +=1

        #wait for the set throttle time (avoids overloading the URL with requests)
        time.sleep(throttle_time)

        #Switch between MapQuest and OSM after a certain number of requests to avoid exceeding temporary usage limits
        #OSM has a lower threshold than MapQuest before they begin temp banning
        if counter == 300 and source == 'OSM':
            msg = "Switching to MapQuest retrieval URL"
            print msg
            logging.info(msg)
            source = 'MQ'
            counter = 0
        if counter == 1500:
            msg = "Switching to OSM retrieval URL"
            print msg
            logging.info(msg)
            source = 'OSM'
            counter = 0

        #Retry the record up to 10 times before moving on to the next record, accounts for temporary network or service issues
        for attempt in range(10):
            #if not the first attempt, wait to allow conditions to improve that caused the first attempt to fail
            if attempt > 0:
                time.sleep(10)
            #if first 5 attempts have failed, switch services and try 5 more attempts with that service
            if attempt == 5:
                if source == 'OSM':
                    source = 'MQ'
                    counter = 0
                else:
                    source = 'OSM'
                    counter = 0
            #set the URL
            if source == 'OSM':
                ##Use OSM reverse retrieval URL
                url = "http://nominatim.openstreetmap.org/reverse?format=json&lat="+str(df['latitude'][idx])+"&lon="+str(df['longitude'][idx])\
                      +"&zoom=18&addressdetails=1&email="+ email_address
            if source == 'MQ':
                ##Use MapQuest reverse retrieval URL
                url = "http://open.mapquestapi.com/nominatim/v1/reverse.php?format=json&lat="+str(df['latitude'][idx])+"&lon="+str(df['longitude'][idx])\
                      +"&zoom=18&addressdetails=1&email="+ email_address
            try:
                data = json.loads(urllib2.urlopen(url).read())

                df['zipcode'][idx] = data['address']['postcode']

                if "neighbourhood" in data['address'].keys():
                    df['neighborhood'][idx] = data['address']['neighbourhood']
                elif "residential" in data['address'].keys():
                    df['neighborhood'][idx] = data['address']['residential']
                elif "suburb" in data['address'].keys():
                    df['neighborhood'][idx] = data['address']['suburb']

                if "road" in data['address'].keys():
                    df['street'][idx] = data['address']['road']
                elif "footway" in data['address'].keys():
                    df['street'][idx] = data['address']['footway']
                elif "path" in data['address'].keys():
                    df['street'][idx] = data['address']['path']
                elif "construction" in data['address'].keys():
                    df['street'][idx] = data['address']['construction']
                elif "pedestrian" in data['address'].keys():
                    df['street'][idx] = data['address']['pedestrian']
                elif "cycleway" in data['address'].keys():
                    df['street'][idx] = data['address']['cycleway']

                if "city" in data['address'].keys():
                    df['city'][idx] = data['address']['city']
                elif "hamlet" in data['address'].keys():
                    df['city'][idx] = data['address']['hamlet']
                elif "suburb" in data['address'].keys():
                    df['city'][idx] = data['address']['suburb']

                msg = "Row %d, lat: %f long: %f -- Successful \n" % (idx, df['latitude'][idx],df['longitude'][idx])
                print msg
                logging.info(msg)
            except urllib2.HTTPError, e:
                error_msg = "Row %d lat: %f long: %f -- HTTP Error: %d , %s \n" % (idx, df['latitude'][idx],df['longitude'][idx], e.code, e.message)
                print error_msg
                logging.warning(error_msg)
                continue
            except urllib2.URLError, e:
                error_msg = "Row %d lat: %f long: %f -- URL error: %s \n" % (idx, df['latitude'][idx],df['longitude'][idx], e.reason.args[1])
                print error_msg
                logging.warning(error_msg)
                continue
            except ValueError, e:
                error_msg = "Row %d lat: %f long: %f -- JSON error: %s \n" % (idx, df['latitude'][idx],df['longitude'][idx], e)
                print error_msg
                logging.warning(error_msg)
                continue
            except KeyError, e:
                error_msg = "Row %d lat: %f long: %f -- Key error: %s \n" % (idx, df['latitude'][idx],df['longitude'][idx], e)
                print error_msg
                logging.warning(error_msg)
                continue
            except e:
                error_msg = "Row %d lat: %f long: %f -- General error: %s \n" % (idx, df['latitude'][idx],df['longitude'][idx], sys.exc_info()[0])
                print error_msg
                logging.warning(error_msg)
                continue
            #if the try worked, break out of the attempt loop
            else:
                break

    #---When all done, save updated dataframe to output file---#
    if delimiter == "":
        df.to_csv(output_file, index=False, encoding='utf-8')
    else:
        df.to_csv(output_file, index=False, delimiter = delimiter, encoding='utf-8')

if __name__ == "__main__":
    main()
