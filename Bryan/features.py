"""
Functions for hand crafting features
"""
__author__ = 'Bryan Gregory'
__email__ = 'bryan.gregory1@gmail.com'
__date__ = '11-19-2013'

#Internal modules
import utils
#Start logger to record all info, warnings, and errors to Logs/logfile.log
log = utils.start_logging(__name__)

#External modules
from sklearn.feature_extraction import text, DictVectorizer
from sklearn import  preprocessing
from scipy.sparse import coo_matrix, hstack, vstack
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn import neighbors

def add(df):
    #-----Add all currently used features to the given dataframe-----#
    hours(df)
    age(df)
    city(df)
    lat_long(df)
    dayofweek(df)
    description_length(df)
    description_length_log(df)
    description_fg(df)
    tagtype_fg(df)
    weekend_fg(df)
    nbr_longlat(df)

def hours(df):
    #-----create hour only, and ~6 hour ranges----#
    df['created_time_hrs'] = [str(x.hour) for x in df['created_time']]
    df['created_time_range'] = '1800-0000'
    for idx in df.index:
        if df['created_time'][idx].hour in [0,1,2,3,4]:
            df['created_time_range'][idx] = '0000-0500'
        if df['created_time'][idx].hour in [5,6,7,8,9,10,11]:
            df['created_time_range'][idx] = '0500-1200'
        if df['created_time'][idx].hour in [12,13,14,15,16,17]:
            df['created_time_range'][idx] = '1200-1800'

def age(df):
    #---Calc age in days---#
    df['age'] = [(datetime.now().date() - x).days for x in df['created_date']]

def city(df):
    df['city'] = 'Chicago'
    for idx in df.index:
        if df['longitude'][idx] < -77 and df['longitude'][idx] > -78:
            df['city'][idx]='Richmond'
        if df['longitude'][idx] < -122 and df['longitude'][idx] > -123:
            df['city'][idx]='Oakland'
        if df['longitude'][idx] < -72 and df['longitude'][idx] > -73:
            df['city'][idx]='New Haven'

def nbr_longlat(df):
    #replace generic neighborhoods with rounded long/lat combination
    df['nbr_longlat'] = df['neighborhood']
    for idx in df.index:
        if df['nbr_longlat'][idx] in ['UNKNOWN','Chicago','Richmond','New Haven','Oakland']:
            df['nbr_longlat'][idx] = df['long_lat_rnd2'][idx]

def lat_long(df):
    df['long_rnd2'] = [str(round(x,2)) for x in df.longitude]
    df['lat_rnd2'] = [str(round(x,2)) for x in df.latitude]
    df['long_lat_rnd2'] = df['long_rnd2'] + df['lat_rnd2']

def dayofweek(df):
    df['dayofweek'] = [str(x.weekday()) for x in df['created_date']]

def weekend_fg(df):
    df['weekend_fg'] = [0 if x < 4 else 1 for x in df['dayofweek']]

#TODO: flag issues created on holidays?  (shameless reuse of feature from past Kaggle contest)
def holidays(df,holidays_list):
    df['holiday_fg'] = 0
    df['holiday_word'] = ''
    for idx in df.index:
        for holiday in holidays_list:
            if holiday.lower() in df.text_all[idx].lower():
                df['holiday_fg'][idx] = 1
                df['holiday_word'][idx] = holiday

def description_length(df):
    #create length of description using linear transform
    df['description_length'] = [-70 if x == '___missing___' else len(x.split()) for x in df.description]

def description_length_log(df):
    #create length of description using log transform
    df['description_length_log'] = [0 if x == '___missing___' else len(x.split()) for x in df.description]
    df['description_length_log'] = np.log(df.description_length_log + 1)

def description_fg(df):
    #flag if no description
    df['description_fg'] = [0 if x == '___missing___' else 1 for x in df.description]

def tagtype_fg(df):
    #flag if no tagtype
    df['tagtype_fg'] = [0 if x == 'na' else 1 for x in df.tag_type]

def vectors(dfTrn, dfTest, model_features):
    #Create feature vectors (matrices) for all text, categorical, or boolean features currently used by model
    for feature in model_features:
    #--------------Text based features section-----------------#
        if feature == 'summary_cat':
            #Vectorize summary into a categorical representation only (not broken down into words or n-grams)
            summary_vec = DictVectorizer().fit([{'feature':value} for value in np.append(dfTrn.summary,dfTest.summary)])
            model_features[feature][0] = summary_vec.transform([{'feature':value} for value in dfTrn.summary])
            model_features[feature][1] = summary_vec.transform([{'feature':value} for value in dfTest.summary])
        if feature == 'summary_count':
            #Word count vector (bag of words) from summary
            count_vec = text.CountVectorizer(min_df = 3, max_df = 0.9, strip_accents = 'unicode', binary = True)
            count_vec.fit(np.append(dfTrn.summary.values,dfTest.summary.values))
            model_features[feature][0] = count_vec.transform(dfTrn.summary.values)
            model_features[feature][1] = count_vec.transform(dfTest.summary.values)
        if feature == 'summary_tfidf_word':
            #Tfidf vector from summary using word analyzer
            tfidf_vec = text.TfidfVectorizer(min_df=6,  max_features=None, strip_accents='unicode',analyzer='word', token_pattern=r'\w{1,}',ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1)
            tfidf_vec.fit(np.append(dfTrn.summary.values,dfTest.summary.values))
            model_features[feature][0] = tfidf_vec.transform((dfTrn.summary.values))
            model_features[feature][1] = tfidf_vec.transform((dfTest.summary.values))
        if feature == 'summary_tfidf_char_wb':
            #Tfidf vector from summary using char_wb analyzer
            tfidf_vec = text.TfidfVectorizer(min_df=3,  max_features=None, strip_accents='unicode',analyzer='char_wb', token_pattern=r'\w{1,}',ngram_range=(2, 30), use_idf=1,smooth_idf=1,sublinear_tf=1)
            tfidf_vec.fit(np.append(dfTrn.summary.values,dfTest.summary.values))
            model_features[feature][0] = tfidf_vec.transform((dfTrn.summary.values))
            model_features[feature][1] = tfidf_vec.transform((dfTest.summary.values))
        if feature == 'summary_descr_tfidf_word':
            #Tfidf vector from description+summary using word analyzer
            tfidf_vec = text.TfidfVectorizer(min_df=6,  max_features=None, strip_accents='unicode',analyzer='word', token_pattern=r'\w{1,}',ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1)
            tfidf_vec.fit(np.append(np.append(dfTrn.summary.values,dfTrn.description.values),np.append(dfTest.summary.values,dfTest.description.values)))
            model_features[feature][0] = tfidf_vec.transform((dfTrn.summary.values))
            model_features[feature][1] = tfidf_vec.transform((dfTest.summary.values))
        if feature == 'summary_descr_tfidf_char_wb':
            #Tfidf vector from description+summary using char_wb analyzer
            tfidf_vec = text.TfidfVectorizer(min_df=3,  max_features=None, strip_accents='unicode',analyzer='char_wb', token_pattern=r'\w{1,}',ngram_range=(2, 30), use_idf=1,smooth_idf=1,sublinear_tf=1)
            tfidf_vec.fit(np.append(np.append(dfTrn.summary.values,dfTrn.description.values),np.append(dfTest.summary.values,dfTest.description.values)))
            model_features[feature][0] = tfidf_vec.transform((dfTrn.summary.values))
            model_features[feature][1] = tfidf_vec.transform((dfTest.summary.values))
    #--------------Categorical features section-----------------#
        if feature == 'city':
            #Vectorize cities
            city_vec = DictVectorizer().fit([{'feature':value} for value in np.append(dfTrn.city,dfTest.city)])
            model_features[feature][0] = city_vec.transform([{'feature':value} for value in dfTrn.city])
            model_features[feature][1] = city_vec.transform([{'feature':value} for value in dfTest.city])
        if feature == 'tag_type':
            #Vectorize tag types
            tagtype_vec = DictVectorizer().fit([{'feature':value} for value in np.append(dfTrn.tag_type,dfTest.tag_type)])
            model_features[feature][0] = tagtype_vec.transform([{'feature':value} for value in dfTrn.tag_type])
            model_features[feature][1] = tagtype_vec.transform([{'feature':value} for value in dfTest.tag_type])
        if feature == 'source':
            #Vectorize sources
            source_vec = DictVectorizer().fit([{'feature':value} for value in np.append(dfTrn.source,dfTest.source)])
            model_features[feature][0] = source_vec.transform([{'feature':value} for value in dfTrn.source])
            model_features[feature][1] = source_vec.transform([{'feature':value} for value in dfTest.source])
        if feature == 'long_lat':
            #Vectorize rounded Long+Lat
            longlat_vec = DictVectorizer().fit([{'feature':value} for value in np.append(dfTrn.long_lat_rnd2,dfTest.long_lat_rnd2)])
            model_features[feature][0] = longlat_vec.transform([{'feature':value} for value in dfTrn.long_lat_rnd2])
            model_features[feature][1] = longlat_vec.transform([{'feature':value} for value in dfTest.long_lat_rnd2])
        if feature == 'neighborhood':
            #Vectorize neighborhoods
            neighborhood_vec = DictVectorizer().fit([{'feature':value} for value in np.append(dfTrn.neighborhood,dfTest.neighborhood)])
            model_features[feature][0] = neighborhood_vec.transform([{'feature':value} for value in dfTrn.neighborhood])
            model_features[feature][1] = neighborhood_vec.transform([{'feature':value} for value in dfTest.neighborhood])
        if feature == 'street':
            #Vectorize street names
            street_vec = DictVectorizer().fit([{'feature':value} for value in np.append(dfTrn.street,dfTest.street)])
            model_features[feature][0] = street_vec.transform([{'feature':value} for value in dfTrn.street])
            model_features[feature][1] = street_vec.transform([{'feature':value} for value in dfTest.neighborhood])
        if feature == 'zipcode':
            #Vectorize zipcodes
            zipcode_vec = DictVectorizer().fit([{'feature':value} for value in np.append(dfTrn.zipcode,dfTest.zipcode)])
            model_features[feature][0] = zipcode_vec.transform([{'feature':value} for value in dfTrn.zipcode])
            model_features[feature][1] = zipcode_vec.transform([{'feature':value} for value in dfTest.zipcode])
        if feature == 'hours_range':
            #Vectorize hours range of creation time
            hrsrange_vec = DictVectorizer().fit([{'feature':value} for value in np.append(dfTrn.created_time_range,dfTest.created_time_range)])
            model_features[feature][0] = hrsrange_vec.transform([{'feature':value} for value in dfTrn.created_time_range])
            model_features[feature][1] = hrsrange_vec.transform([{'feature':value} for value in dfTest.created_time_range])
        if feature == 'day_of_week':
            #Vectorize day of week
            dayofweek_vec = DictVectorizer().fit([{'feature':value} for value in np.append(dfTrn.dayofweek,dfTest.dayofweek)])
            model_features[feature][0] = dayofweek_vec.transform([{'feature':value} for value in dfTrn.dayofweek])
            model_features[feature][1] = dayofweek_vec.transform([{'feature':value} for value in dfTest.dayofweek])
    #--------------Binary/Boolean features section-----------------#
        if feature == 'description_fg':
            #Description flag
            model_features[feature][0] = dfTrn.ix[:,['description_fg']].as_matrix()
            model_features[feature][1] = dfTest.ix[:,['description_fg']].as_matrix()
        if feature == 'weekend_fg':
            #Weekend flag
            model_features[feature][0] = dfTrn.ix[:,['weekend_fg']].as_matrix()
            model_features[feature][1] = dfTest.ix[:,['weekend_fg']].as_matrix()
        if feature == 'tagtype_fg':
            #Tagtype flag
            model_features[feature][0] = dfTrn.ix[:,['tagtype_fg']].as_matrix()
            model_features[feature][1] = dfTest.ix[:,['tagtype_fg']].as_matrix()
    #---------------Unused/obsolete features------------------------#
        if feature == 'hours':
            #Vectorize all hours of creation time
            hrs_vec = DictVectorizer().fit([{'feature':value} for value in np.append(dfTrn.created_time_hrs,dfTest.created_time_hrs)])
            model_features[feature][0] = source_vec.transform([{'feature':value} for value in dfTrn.created_time_hrs])
            model_features[feature][1] = source_vec.transform([{'feature':value} for value in dfTest.created_time_hrs])

def numerical(dfTrn, dfTest, model_features):
    #Transform, scale, and/or standardize all numerical features currently used by model, store as matrix
    for feature in model_features:
        if feature == 'description_length':
            #linear description length
            model_features[feature][0] = standardize(dfTrn,[feature])
            model_features[feature][1] = standardize(dfTest,[feature])
        if feature == 'description_length_log':
            model_features[feature][0] = standardize(dfTrn,[feature])
            model_features[feature][1] = standardize(dfTest,[feature])
        if feature == 'tot_income':
            model_features[feature][0] = standardize(dfTrn,[feature])
            model_features[feature][1] = standardize(dfTest,[feature])
        if feature == 'est_pop':
            model_features[feature][0] = standardize(dfTrn,[feature])
            model_features[feature][1] = standardize(dfTest,[feature])
        if feature == 'avg_income':
            model_features[feature][0] = standardize(dfTrn,[feature])
            model_features[feature][1] = standardize(dfTest,[feature])

def sub_feature(df,feature1, feature2, values):
    """Substitute one list of features for another, within a given subset.
    feature1 = feature to replace feature2
    feature2 = feature to be replaced
    values = values of feature2 for which when true, sub in the corresponding feature1
    """
    for idx in df.index:
        if df[feature2][idx] in values:
            df[feature2][idx] = str(df[feature1][idx])

def knn_thresholding(df, column, threshold=15, k=3):
    """Cluster rare samples in data[column] with frequency less than
    threshold with one of k-nearest clusters

    parameters:
       data - pandas.DataFrame containing colums: latitude, longitude, column
       column - the name of the column to threshold
       threshold - the minimum sample frequency
       k - the number of k-neighbors to explore when selecting cluster partner
    """
    def ids_centers_sizes(data):
        dat = np.array([(i, data.latitude[data[column]==i].mean(),
                        data.longitude[data[column]==i].mean(),
                        (data[column]==i).sum())
                        for i in set(list(data[column]))])
        return dat[:,0], dat[:,1:-1].astype(float), dat[:,-1].astype(int)

    knn = neighbors.NearestNeighbors(n_neighbors=k)
    while True:
        ids, centers, sizes = ids_centers_sizes(df)
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
                df[column][df[column]==cids[i]] = cids[sel]
                cids[cids==i] = cids[sel]
                sizes[i] = total_size
                sizes[sel] = total_size

    return df

def standardize(df,features):
    #---------------------------------------------------------------------
    #Standardize list of quant features (remove mean and scale to unit variance)
    #---------------------------------------------------------------------
    scaler = preprocessing.StandardScaler()
    if features == 'all':
        mtx = scaler.fit_transform(df.as_matrix())
    else:
        mtx = scaler.fit_transform(df.ix[:,features].as_matrix())
    return mtx