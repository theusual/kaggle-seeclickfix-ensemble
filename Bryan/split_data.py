'''
Split a data set into two randomly, line by line.  Used to create a hold out data set from a training set for cross-validation
Usage: split.py <input whole training set> <output split training set> <output hold out test set> <headers? (Y/N)> [% to holdout] [SEED]
Ex.: python split_data.py Data/text_body_trn.svm Data/text_body_trn_split.svm Data/text_body_trn_holdout.svm N .1 888
'''
__author__ = 'Bryan Gregory'
__email__ = 'bryan.gregory1@gmail.com'
__date__ = '09-05-2013'


import csv
import sys
import random

print __doc__

input_file = sys.argv[1]
output_file1 = sys.argv[2]
output_file2 = sys.argv[3]
headers_fg = sys.argv[4]

i = open( input_file )
o1 = open( output_file1, 'wb' )
o2 = open( output_file2, 'wb' )

if (headers_fg == 'Y'):
    headers = i.next()
    o1.write( headers )
    o2.write( headers )

try:
	P = 1 - float(sys.argv[5])
except IndexError:
	P = 0.15

try:
	seed = sys.argv[6]
except IndexError:
	seed = None

print "Splitting %s percent of data" % (P*100)

if seed:
	random.seed(seed)

counter = 0

for line in i:
	r = random.random()
	if r < P:
		o2.write( line )
	else:
		o1.write( line )
	counter += 1
print "Split complete.  Total rows processed: {:,}".format(counter)
