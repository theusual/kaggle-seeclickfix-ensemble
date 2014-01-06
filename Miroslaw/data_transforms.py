import config, utils

import pandas as pd
import numpy as np

from scipy import sparse
from matplotlib.mlab import find
from datetime import datetime
from itertools import combinations
from sklearn import feature_extraction, preprocessing

## Global Variables ##
id = ['id']
features = ['latitude', 'longitude', 'summary', 'description', 
            'source', 'created_time', 'tag_type']
geocols = ['city', 'zipcode', 'neighborhood']
labels = ['num_views', 'num_votes', 'num_comments']
possibly_missing = ['description', 'source', 'tag_type']

def weighted_average(p1, p2, weights, rmse=True):
    """
    Computes a weighted average of p1 and p2
    
    Args:
        p1: first prediction (n_samples, n_targets)
        p2: second prediction (n_samples, n_targets)
        weights: the weights of the average with respect to p1, each weight must be
                  in the interval [0, 1]
                  weights can be either a float, or numpy array with 
                  shape (n_targets, ) or (n_samples, 1) or (n_samples, n_targets)
        rmse: boolean, if True average in rmse-space instead of rmsle-space
    """
    if rmse:
        p1 = np.log(p1 + 1)
        p2 = np.log(p2 + 1)
    avg = p1 * weights + (1 - weights) * p2
    if rmse:
        avg = exp(avg) - 1
    return avg 

def create_weight(segments, weight_dict):
    """
    create a weight matrix for averaging given a list of segments
    and a dict containing weights for each segment type
    """
    weights = [weight_dict[s] for s in segments]
    return np.array(weights)
    
## Transformation functions ##
# These should have the form func_name(*args, **kwargs)
def ngramBOW(data, cols=None, min_gram=1, max_gram=1, min_df=3, max_df=1.0, 
             lowercase=True, strip_accents='unicode', stop_words=None, 
             token_pattern=u'(?u)\b\w\w+\b'):
    """
    Creates a bag of words representation of data
    
    Args:
        data: iterable 
              an iterable of documents containing text elements
        max_gram: int
                  the minimum ngram range to use
        max_gram: int
                  the maximum ngram range to use
        min_df: float in range [0.0, 1.0] or int
                When float, will only consider words that occur proportionally
                in at least min_df documents. When int, will only consider words 
                that occur at least min_df times in a document
        max_df: float in range [0.0, 1.0] or int
                When float, will only consider words that occur proportionally
                in at most max_df documents. When int, will only consider words 
                that occur at most max_df times in a document
        strip_accents: None, unicode, or ascii
                       Remove accents during the preprocessing step
        lowercase: boolean
                   Sets all words to lowercase if True
        stop_words: None, 'english', or list
                    If None, no stopword removal is performed
                    If english, use stopwards from the english stopword corpus
                    if list, consider words in list as stopwords
    
    Returns:
        A bag of words representation of data as a scipy.sparse matrix
    """
    count_vect = feature_extraction.text.CountVectorizer(min_df=min_df, 
                    max_df=max_df, ngram_range=(min_gram,max_gram), binary=True,
                    lowercase=lowercase, stop_words=stop_words, 
                    strip_accents=strip_accents, token_pattern=token_pattern,
                    charset_error='replace')
    return count_vect.fit_transform(data)

def ngramTFIDF(data, cols=None, min_gram=1, max_gram=1, min_df=3, max_df=1.0, stop_words=None, 
               strip_accents='unicode', lowercase=True, sublinear_tf=True,
               token_pattern=u'(?u)\b\w\w+\b'):
    """
    Creates a TFIDF representation of data
    
    Args:
        data: iterable 
              an iterable of documents containing text elements
        max_gram: int
                  the maximum ngram range to use
        min_df: float in range [0.0, 1.0] or int
                When float, will only consider words that occur proportionally
                in at least min_df documents. When int, will only consider words 
                that occur at least min_df times in a document
        max_df: float in range [0.0, 1.0] or int
                When float, will only consider words that occur proportionally
                in at most max_df documents. When int, will only consider words 
                that occur at most max_df times in a document
        strip_accents: None, unicode, or ascii
                       Remove accents during the preprocessing step
        lowercase: boolean
                   Sets all words to lowercase if True
        stop_words: None, 'english', or list
                    If None, no stopword removal is performed
                    If english, use stopwards from the english stopword corpus
                    if list, consider words in list as stopwords
        sublinear_tf: boolean
                      If True, compute term frequencies using log(tf + 1)
                      If False, use raw term frequencies
    Returns:
        A bag of words representation of data as a scipy.sparse matrix
    """
    tfidf_vect = feature_extraction.text.TfidfVectorizer(min_df=min_df, 
                    max_df=max_df, ngram_range=(min_gram,max_gram), lowercase=lowercase, 
                    stop_words=stop_words, strip_accents=strip_accents, 
                    sublinear_tf=sublinear_tf, token_pattern=token_pattern,
                    charset_error='ignore')
    return tfidf_vect.fit_transform(data)

def one_hot_encoder(data, min_freq=3):
    """
    Returns a one hot encoding of data
    
    Args:
        data: numpy.array of rank 1 containing categorical elements
        min_freq: int
                  the minimum number of times a category must appear to get
                  its own category. Dumped into a 'rare' category otherwise 
    
    Returns:
        A binary encoding of all categories in data
    """
    ohe = preprocessing.OneHotEncoder()
    numerical_data = integerizer(group_rares(data, min_freq))
    numerical_data.shape = (numerical_data.shape[0], 1)
    return ohe.fit_transform(numerical_data)

def integerizer(data):
    """
    Converts each individual data point in data into a numerical value.
    Useful for preprocessing data to be passed into sklearn's OneHotEncoder
    
    Args:
        data - an iterable containing hashable elements
    
    Returns:
        a numpy array with each distinct datapoint in data converted to a
        distinct integer
    """
    int_map = {}
    new_data = []
    curr_id = 0
    for pt in data:
        if pt not in int_map:
            int_map[pt] = curr_id
            curr_id += 1
        new_data.append(int_map[pt])
    return np.array(new_data)

def group_rares(data, cutoff=3):
    """
    Reclasses rare values in data to a single class named '__RARE__'
    
    Args:
        data - an iterable containing hashable elements
        cutoff - an integer upper bound for the number of occurances a particular
                 class must appear in data for it to be considered rare
    
    Returns:
        a numpy array with all rare elements relabeled to be members of the
        '__RARE__' class
    """
    counts = {}
    new_data = []
    for pt in data:
        counts[pt] = counts.get(pt, 0) + 1
    for pt in data:
        if counts[pt] <= cutoff:
            new_data.append('__RARE__')
        else:
            new_data.append(pt)
    return np.array(new_data)
    
def get_train(data):
    """return a dataframe containing training data"""
    data = pd.read_csv(config.TRAINFILE)
    geodata = pd.read_csv(config.GEODATA)
    if config.USE_BRYANS_DATA:
        bdata = pd.read_csv(config.BRYAN_TRAIN)
        old_index = bdata.index
        bdata.index = bdata.id
        bdata = bdata.ix[data.id]
        bdata.index = old_index
        bdata.created_time = data.created_time
        data = bdata
    n_train = data.shape[0]
    data = data[id + features]
    for col in geocols:
        data[col] = np.array(geodata[col][:n_train])
    return data
    
def get_test(data):
    """return a dataframe containing test data"""
    data = pd.read_csv(config.TESTFILE)
    geodata = pd.read_csv(config.GEODATA)
    if config.USE_BRYANS_DATA:
        bdata = pd.read_csv(config.BRYAN_TEST)
        old_index = bdata.index
        bdata.index = bdata.id
        bdata = bdata.ix[data.id]
        bdata.index = old_index
        bdata.created_time = data.created_time
        data = bdata
    n_test = data.shape[0]
    n_geo = geodata.shape[0]
    data = data[id + features]
    for col in geocols:
        data[col] = np.array(geodata[col][n_geo-n_test:])
    return data
    
def get_targets(data):
    """return an array containing train labels with a log(y + 1) transform"""
    data = pd.read_csv(config.TRAINFILE)
    y = data[labels]
    return np.array(y)
    
def get_ids(data, **kwargs):
    return np.array(data.id)

def text_vectorizer(data, cols, method='bow', **args):
    """
    Wrapper for ngramBOW and ngramTFIDF that can be applied to multiple columns 
    of a pandas DataFrame
    
    Args:
        data - a pandas DataFrame
        cols - a list of colum names on which to run vectorization
        method - 'bow' or 'tfidf' 
        **args - args to pass to vectorizor
    
    Returns:
        A scipy sparse CSR matrix concatentaed for each column vectorization 
        from data
    """
    if method == 'bow':
        vectorizer = ngramBOW
    elif method == 'tfidf':
        vectorizer = ngramTFIDF
    else:
        raise ValueError('Unkown vectorization method %s' % method)
    
    return sparse.hstack([vectorizer(data[col], **args) 
                           for col in cols]).tocsr()

def columnOHE(data, column, **args):
    """
    Wrapper to ohe_hot_encoder that takes a pandas dataframe and performs
    one hot encoding on a specific column
    """
    return one_hot_encoder(data[column], **args)
    
def hstack(datasets):
    """
    Wrapper for hstack method from numpy or scipy. If any of the input
    datasets are sparse, then sparse.hstack is used instead of np.hstack
    
    Args:
        datasets: datasets to be stacked, all must have same shape along axis 0
    
    Returns:
        a sparse CSR matrix if any dataset in datasets is sparse
        a numpy array otherwise
    """
    if np.any([sparse.issparse(d) for d in datasets]):
        stack = lambda x: sparse.hstack(x).tocsr()
    else:
        stack = np.hstack
    return stack(datasets)

def drop_disjoint(train, test):
    """
    Removes all features from train and test that are strictly all 0 in either the train
    set or test set
    
    train, and test are scipy.sparse CSR matricies 
    """
    in_training = find(train.sum(0).A.flatten() != 0)
    train = train[:, in_training]
    test = test[:, in_training]
    in_test = find(test.sum(0).A.flatten() != 0)
    train = train[:, in_test]
    test = test[:, in_test]
    return sparse.vstack((train, test)).tocsr()
    
def reduce_data(data, keep_n):
    """
    Return only the final keep_n elements of the data
    
    Args:
        data - an iterable that supports indexing
        keep_n - an integer for the number of elements to keep
    
    Returns:
        data[-keep_n:]
    """
    return data[-keep_n:]
    
def sparse_reduce_and_filter(data, keep_n_train=None):
    """
    A wrapper function that calls reduce_data on training data and then
    drop_disjoint on the training and test data
    
    Args:
        data - a 2-tuple or iterable of length 2 
               data[0] is a sparse matrix contains training data followed by test data
               data[1] is an array containing the training lables for the test data
        keep_n_train - None or int
                       if None, all training data is returned
                       if int, only the last keep_n_train data points are returned
    
    Returns:
        A scipy sparse matrix containing keep_n_train points from the train set
        followed by the entire test set. This matrix is filered with the
        drop_disjoint function
    """
    data, targets = data
    n_train = targets.shape[0]
    data_test = data[n_train:]
    data_train = data[:n_train]
    if keep_n_train is not None:
        data_train = reduce_data(data_train, keep_n_train)
        targets = reduce_data(targets, keep_n_train)
    return drop_disjoint(data_train, data_test)

def get_df_cols(data, cols):
    """
    A wrapper that extracts cols from a pandas dataframe
    
    Args:
        data - a pandas DataFrame
        cols - a list of column names that will be extracted from data
    
    Returns:
        data[cols]
    """
    return data[cols]

def concat_df(df_list, ignore_index=False):
    """
    A wrapper for pandas concat function
    
    Args:
        df_list - a list of pandas DataFrames
        ignore_index - a boolean 
                       controls if the original row-index of the dataframes is
                       perserved 
    
    Returns:
        A concatenated DataFrame created from df_list
    """
    return pd.concat(df_list, ignore_index=ignore_index)

def fillna(df, cols=None, fill_val='__missing__'):
    """
    A wrapper for pandas fillna function
    
    Args:
        df - a pandas DataFrame
        cols - None or list of column names
               if None, fillna is called on the entire DataFrame
               if a list of columns, fillna is called only on those columns
        fill_val - a value used to fill in any NA elements
    
    Returns:
        a DataFrame with NA values replaced with fill_val
    """
    if cols is None:
        df = df.fillna(fill_val)
    else:
        df[cols] = df[cols].fillna(fill_val)
    return df

def col2datetime(df, col, datestr='%m/%d/%Y %H:%M'):
    """
    Converts date strings in a DataFrame to datetime objects
    
    Args:
        df - a pandas DataFrame
        col - the column name in df that contains the date strings
        datestr - a string that encodes the formatting of the datetime strings
                  see python's documentation for datetime to see how these
                  strings are formatted
    
    Returns:
        The DataFrame with all date strings in df[col] converted to datetime objects
    """
    for idx in df.index:
        d = df[col][idx]
        df[col][idx] = datetime.strptime(d, datestr)
    return df

def replace_col_val(data, column, repcol, val):
    """
    Replaces any values val in a DataFrame column with the value from column repcol
    Currently both columns are converted to strings ... not really wise
    
    Args:
        data - a pandas DataFrame
        column - the name of the column where values will be replaced
        repcol - the name of the column where replacement values are taken from
        val - a string that is used to match against the column
    
    Returns:
        a DataFrame with the condition:
            if data[column][i] == val then data[column][i] = data[repcol][i]
    """
    new_col = []
    data[column] = data[column].astype('S100')
    data[repcol] = data[repcol].astype('S100')
    data[column][data[column]==val] = data[repcol][data[column]==val]
    return data
    
def knn_threshold(data, column, threshold=15, k=3):
    """
    Cluster rare samples in data[column] with frequency less than 
    threshold with one of k-nearest clusters 
    
    Args:
        data - pandas.DataFrame containing colums: latitude, longitude, column
        column - the name of the column to threshold
        threshold - the minimum sample frequency
        k - the number of k-neighbors to explore when selecting cluster partner
    """
    from sklearn import neighbors
    
    def ids_centers_sizes(data):
        dat = np.array([(i, data.latitude[data[column]==i].mean(), 
                        data.longitude[data[column]==i].mean(),
                        (data[column]==i).sum()) 
                        for i in set(list(data[column]))])
        return dat[:,0], dat[:,1:-1].astype(float), dat[:,-1].astype(int)

    knn = neighbors.NearestNeighbors(n_neighbors=k)
    while True:
        ids, centers, sizes = ids_centers_sizes(data)
        asrt = np.argsort(sizes)
        
        if sizes[asrt[0]] >= threshold:
            break
             
        cids = np.copy(ids)
        knn.fit(centers)
        
        for i in asrt:
            if sizes[i] < threshold:
                nearest = knn.kneighbors(centers[i])[1].flatten()
                nearest = nearest[nearest != i]
                sel = nearest[np.argmin(sizes[nearest])]
                total_size = sizes[sel] + sizes[i]
                data[column][data[column]==cids[i]] = cids[sel]
                cids[cids==i] = cids[sel]
                sizes[i] = total_size
                sizes[sel] = total_size
                
    return data
    
def is_weekend(data):
    """
    Adds a binary is_weekend column to a pandas DataFrame 
    
    Args:
        data - a pandas DataFrame containing a 'created_time' column of
               datetime objects
    Returns:
        A DataFrame with an additional 'is_weekend' column
        if data['created_time'][i] is Friday, Saturday or Sunday then
        data['is_weekend'][i] = 1
    """
    data['is_weekend'] = 0
    weekend_days = {4, 5, 6}
    for idx in data.index:
        wd = data.created_time[idx].weekday()
        if wd in weekend_days:
            data['is_weekend'][idx] = 1
    return data

def time_of_day(data):
    """
    Adds a time_of_day column to a pandas DataFrame
    
    Args:
        data - a pandas DataFrame containing a 'created_time' column of 
               datetime objects
    
    Returns:
        A DataFrame with an additional 'time_of_day' column
        time_of_day is based on the hour and the day is broken up into 6 4h segments
        each segment is given it's own integer class 0-5
    """
    data['time_of_day'] = 0
    for idx in data.index:
        h = data.created_time[idx].hour
        if h < 4:
            data['time_of_day'][idx] = 0
        elif h < 8:
            data['time_of_day'][idx] = 1
        elif h < 12:
            data['time_of_day'][idx] = 2
        elif h < 16:
            data['time_of_day'][idx] = 3
        elif h < 20:
            data['time_of_day'][idx] = 4
        elif h < 24:
            data['time_of_day'][idx] = 5
    return data

def city_neighborhood(data):
    """
    A helper method that concatenates the city name with neighborhood name
    This is used as an insurance policy to distinguish two cities that may 
    share the same neighborhood name
    
    Args:
        data - a pandas DataFrame with city and neighborhood columns
    
    Returns:
        a DataFrame with a new column 'city_neighborhood' that contains the
        concatenated string data.city[i] + ' ' + data.neighborhood[i]
    """
    data['city_neighborhood'] = ''
    for idx in data.index:
        c = data.city[idx]
        n = data.neighborhood[idx]
        data['city_neighborhood'][idx] = '%s %s' %(c, n)
    return data

def rare_category_replacement(data, column, replace_col, cutoff=3):
    """
    Replace any categorical values that occur less than cutoff times 
    in data[column] with data[replace_col]
    
    Args:
        data - a pandas DataFrame
        column - the name of the column that is being thresholded
        replace_col - the name of the column that is used for replacement values
        cutoff - the cutoff threshold for replacement
    
    Returns:
        a DataFrame that has any values in data[column] that occur less than
        cutoff time replaced with data[replace_col]
    """
    counts = {}
    data[column] = data[column].astype('S100')
    data[replace_col] = data[replace_col].astype('S100')
    for idx in data.index:
        counts[data[column][idx]] = counts.get(data[column][idx], 0) + 1
    
    for idx in data.index:
        if counts[data[column][idx]] < cutoff:
            data[column][idx] = data[replace_col][idx]
    return data
    
def description_length(data, log_transform=True):
    """
    Create a new feature description_length based on the number of words in
    the issue description. Any issues with description '__missing__' are assumed
    to have length 0
    
    Args:
        data - A pandas DataFrame containing a column named 'description'
        log_transform - boolean
                        if True data[description_length] = log(# of words + 1)
                        if False data[description_length] = # of words 
    Returns:
        a DataFrame with an addition column 'description_length' 
    """
    data['description_length'] = 0
    for idx in data.index:
        d = data.description[idx]
        if d == '__missing__':
            data.description_length[idx] = 0
        else:
            data.description_length[idx] = len(d.split())
    
    if log_transform:
        data.description_length = np.log(data.description_length + 1)
        
    return data

def group_data(data, cols, degree=3):
    """
    Create higher-order combinations of categorial features
    
    Args:
        data - a pandas DataFrame
        cols - a list of column names that will be used for creating higher-order
               groupings
        degree - an integer identifying the order of the combinations
    
    Returns:
        a DataFrame with additional colums for each of the higher-order column
        combination. The names of the new columns are concatenated strings
        of the column names used in the combination separated by an '_'
        
        Eg: 2nd degree combinations of data['col1'] and data['col2'] are located
            in data['col1_col2']
    """
    new_data = []
    m,n = data[cols].shape
    for indices in combinations(range(n), degree):
        group_ids = data.groupby( \
        list(data[cols].columns[list(indices)])) \
        .grouper.group_info[0]
        new_data.append(group_ids)
    data['_'.join(cols)] = np.array(new_data).flatten()
    return data

def create_city_and_api_groups(data):
    """
    Creates a list of groups based on city name and issue source used in scaling
    the final predictions. 
    
    Args: 
        data - a pandas dataframe that contains city and source columns
    
    Returns:
        a list of strings, each string contains the city name along with a '_RAC'
        suffix if the issue source is 'remote_api_created'
    """
    groups = []
    for idx in data.index:
        s = data.city[idx]
        if data.source[idx] == 'remote_api_created':
            s = s + ' RAC'
        groups.append(s)
    return groups
    
        
