
__author__ = 'Bryan Gregory'
__email__ = 'bryan.gregory1@gmail.com'
__date__ = '12-24-2013'

#Internal modules
import utils
#Start logger to record all info, warnings, and errors to Logs/logfile.log
log = utils.start_logging(__name__)
import munge
import train
import data_io
import features
import ensembles

#External modules
import sys
import pandas as pd
from datetime import datetime

def main():
    log.info('********New program instance started********')

    #-------------Load Environment----------------------#
    #Get program settings and model settings from SETTINGS.json file in root directory
    settings, model_settings = utils.load_settings()

    #If not using cached data, then load raw data, clean/munge it, create hand-crafted features, slice it for CV
    if settings['use_cached_data'] == 'y':
        log.info('==========LOADING CACHED FEATURES===========')
        dfTrn = data_io.load_cached_object('dfTrn')
        dfTest = data_io.load_cached_object('dfTest')
        dfCV = data_io.load_flatfile_to_df('Data/CV.csv')
    else:
        #-------Data Loading/Cleaning/Munging------------#
        #Load the data
        log.info('===============LOADING DATA=================')
        dfTrn = data_io.load_flatfile_to_df(settings['file_data_train'])
        dfTest = data_io.load_flatfile_to_df(settings['file_data_test'])
        dfCV = data_io.load_flatfile_to_df('Data/CV.csv')

        #Clean/Munge the data
        log.info('=======CLEANING AND MUNGING DATA============')
        dfTrn = munge.clean(dfTrn)
        dfTest = munge.clean(dfTest)

        #-------Feature creation-------------------------#
        #Add all currently used hand crafted features to dataframes
        log.info('====CREATING HAND-CRAFTED DATA FEATURES=====')
        features.add(dfTrn)
        features.add(dfTest)

        #---------Data slicing/parsing--------------------------#
        #Split data for CV
        if settings['generate_cv_score'] == 'y':
            log.info('=====SPLITTING DATA FOR CROSS-VALIDATION====')
            if settings['cv_method'] == 'april':
                dfTrnCV, dfTestCV = munge.temporal_split(dfTrn, (2013, 04, 1))
            elif settings['cv_method'] == 'march':
                #take an addtional week from February b/c of lack of remote_api source issues in March
                dfTrnCV, dfTestCV = munge.temporal_split(dfTrn, (2013, 02, 21))
            elif settings['cv_method'] == 'list_split':
                #load stored list of data points and use those for CV
                dfCVlist = pd.DataFrame({'id': data_io.load_cached_object('Cache/cv_issue_ids.pkl'), 'dummy': 0})
                dfTrnCV, dfTestCV = munge.list_split(dfTrn, dfCVlist)

    #--------------Modeling-------------------------#
    #If cached models exist then load them for reuse into segment_models.  Then run through model_settings and for
    # each model where 'use_cached_model' is false then clear the cached model and recreate it fresh
    log.info('=========LOADING CACHED MODELS==============')
    segment_models = data_io.load_cached_object('segment_models')
    if segment_models == None:
        log.info('=========CACHED MODELS NOT LOADED===========')
        for model in model_settings:
            model['use_cached_model'] = 'n'
        segment_models = []
    #Initialize new model for models not set to use cache
    log.info('=======INITIALIZING UN-CACHED MODELS========')
    index = 0
    for model in model_settings:
        if model_settings[model]['use_cached_model'] == 'n':
            new_model = ensembles.Model(model_name=model,target=model_settings[model]['target'],
                                        segment=model_settings[model]['segment'],
                                        estimator_class=model_settings[model]['estimator_class'],
                                        estimator_params=model_settings[model]['estimator_params'],
                                        features=model_settings[model]['features'],
                                        postprocess_scalar=model_settings[model]['postprocess_scalar'])
            #Flag the model as not cached, so that it does not get skipped when running the modeling process
            new_model.use_cached_model='n'
            #Project specific model attributes not part of base class
            new_model.KNN_neighborhood_threshold=model_settings[model]['KNN_neighborhood_threshold']
            new_model.sub_zip_neighborhood=model_settings[model]['sub_zip_neighborhood']
            segment_models[index] = new_model
            log.info('Model %s intialized at index %i' % (model,index))
        index += 1

    #Cross validate all segment models (optional)
    if settings['export_cv_predictions_all_models'] == 'y' or settings['export_cv_predictions_new_models'] == 'y':
        log.info('============CROSS VALIDATION================')
        for model in segment_models[:]:
            #If model has cached CV predictions then skip predicting and just export them (if selected in settings)
            if hasattr(model,'dfCVPredictions'):
                log.info('Cached CV predictions found.  Using cached CV predictions.')
                if settings['export_cv_predictions_all_models'] == 'y':
                    data_io.save_predictions(model.dfCVPredictions,model.target,model_name=model.model_name,
                                             directory=settings['dir_submissions'],
                                             estimator_class=model.estimator_class, note='CV_list')
            else:
                print_model_header(model)
                #Prepare segment model:  segment and create feature vectors for the CV data set
                dfTrn_Segment, dfTest_Segment = prepare_segment_model(dfTrnCV,dfTestCV,model)
                #Generate CV predictions
                train.cross_validate(model, settings, dfTrn_Segment, dfTest_Segment)
                #Cache the CV predictions as a dataframe stored in each segment model
                model.dfCVPredictions = dfTest_Segment.ix[:,['id',model.target]]
                if settings['export_cv_predictions_new_models'] == 'y':
                    data_io.save_predictions(model.dfCVPredictions,model.target,model_name=model.model_name,
                                             directory=settings['dir_submissions'],
                                             estimator_class=model.estimator_class, note='CV_list')

    #Generate predictions on test set for all segment models (optional)
    if settings['export_predictions_all_models'] == 'y' or settings['export_predictions_new_models'] == 'y'\
        or settings['export_predictions_total'] == 'y':
        log.info('=======GENERATING TEST PREDICTIONS==========')
        for model in segment_models[:]:
            #If model has cached test predictions then skip predicting and just export them (if selected in settings)
            if hasattr(model,'dfPredictions'):
                log.info('Cached test predictions found for model %s.  Using cached predictions.' % model.model_name)
                if settings['export_predictions_all_models'] == 'y':
                    data_io.save_predictions(model.dfPredictions,model.target,model_name=model.model_name,
                             directory=settings['dir_submissions'],
                             estimator_class=model.estimator_class,note='TESTset')
            else:
                print_model_header(model)
                #Prepare segment model:  segment and create feature vectors for the full TEST data set
                dfTrn_Segment, dfTest_Segment = prepare_segment_model(dfTrn,dfTest,model)
                #Generate TEST set predictions
                model.predict(dfTrn_Segment, dfTest_Segment)
                if settings['export_predictions_all_models'] == 'y' or settings['export_predictions_new_models'] == 'y':
                    data_io.save_predictions(model.dfPredictions,model.target,model_name=model.model_name,
                                             directory=settings['dir_submissions'],
                                             estimator_class=model.estimator_class,note='TESTset')
                log.info(utils.line_break())

    #Cache the trained models and predictions to file (optional)
    if settings['export_cached_models'] == 'y':
        log.info('==========EXPORTING CACHED MODELS===========')
        data_io.save_cached_object(segment_models,'segment_models')

    #Merge each segment model's CV predictions into a master dataframe and export it (optional)----#
    if settings['export_cv_predictions_total'] == 'y':
        log.info('====MERGING CV PREDICTIONS FROM SEGMENTS====')
        dfTestPredictionsTotal = merge_segment_predictions(segment_models, dfTestCV, cv=True)
        #---Apply post process rules to master dataframe---#
        #Set all votes and comments for remote_api segment to 1 and 0
        dfTestPredictionsTotal = dfTestPredictionsTotal.merge(dfTest.ix[:][['source','id']], on='id', how='left')
        for x in dfTestPredictionsTotal.index:
            if dfTestPredictionsTotal.source[x] == 'remote_api_created':
                dfTestPredictionsTotal.num_votes[x] = 1
                dfTestPredictionsTotal.num_comments[x] = 0
        #Export
        timestamp = datetime.now().strftime('%m-%d-%y_%H%M')
        filename = 'Submits/'+timestamp+'--bryan_CV_predictions.csv'
        dfTestPredictionsTotal.to_csv(filename)


    #Merge each segment model's TEST predictions into a master dataframe and export it (optional)----#
    if settings['export_predictions_total'] == 'y':
        log.info('===MERGING TEST PREDICTIONS FROM SEGMENTS===')
        dfTestPredictionsTotal = merge_segment_predictions(segment_models, dfTest)
        #---Apply post process rules to master dataframe---#
        #Set all votes and comments for remote_api segment to 1 and 0
        dfTestPredictionsTotal = dfTestPredictionsTotal.merge(dfTest.ix[:][['source','id']], on='id', how='left')
        for x in dfTestPredictionsTotal.index:
            if dfTestPredictionsTotal.source[x] == 'remote_api_created':
                dfTestPredictionsTotal.num_votes[x] = 1
                dfTestPredictionsTotal.num_comments[x] = 0
        del dfTestPredictionsTotal['source']
        #Export
        filename = 'bryan_test_predictions.csv'
        data_io.save_combined_predictions(dfTestPredictionsTotal, settings['dir_submissions'], filename)

    #End main
    log.info('********Program ran successfully. Exiting********')

##########################################################################################################
def prepare_segment_model(dfTrn,dfTest,model):
    """Given a segment model, create the data segment for that model, then create the feature values for that model
    """
    #Segment the data
    dfTrn_Segment, dfTest_Segment = munge.segment_data(dfTrn, dfTest, model.segment)
    #Apply model-specific neighborhood subbing if enabled, and then apply KNN neighborhood thresholding
    if int(model.KNN_neighborhood_threshold) > 0:
        if model.sub_zip_neighborhood == 'y':
            #Substitute zipcodes for overly common neighborhoods to provide more geographic detail
            log.info('==USING ZIP FOR PLACEHOLDER NEIGHBORHOODS===')
            features.sub_feature(dfTrn_Segment,'zipcode','neighborhood',
                                 ['Richmond','Oakland','Manchester','Chicago','New Haven'])
            features.sub_feature(dfTest_Segment,'zipcode','neighborhood',
                                 ['Richmond','Oakland','Manchester','Chicago','New Haven'])
        log.info('==KNN ON RARE NEIGHBORHOODS WITH COUNT < %i==' % int(model.KNN_neighborhood_threshold))
        dfTrn_Segment = features.knn_thresholding(dfTrn_Segment,'neighborhood',
                                                  int(model.KNN_neighborhood_threshold))
        dfTest_Segment = features.knn_thresholding(dfTest_Segment,'neighborhood',
                                                   int(model.KNN_neighborhood_threshold))
    return dfTrn_Segment, dfTest_Segment

##########################################################################################################
def print_model_header(model):
    """Print header with model info
    """
    features_list = (map(str,model.features.keys()))
    features_list.sort()
    log.info(utils.line_break())
    log.info('MODEL: %s    SEGMENT: %s     TARGET: %s '  % (model.model_name, model.segment, model.target))
    log.info('FEATURES: %s' % features_list)
    log.info('ESTIMATOR CLASS: %s ' % model.estimator)
    log.info('POST-PROCESS SCALAR: %s ' % model.postprocess_scalar)

##########################################################################################################
def merge_segment_predictions(segment_models, dfTest, cv=False):
    """Combine the predictions of all segment models into a master file
    """
    all_predictions = {}
    for model in segment_models[:]:
        if model.target not in all_predictions.keys():
            if cv == False:
                all_predictions[model.target] = model.dfPredictions.ix[:]
            else:
                all_predictions[model.target] = model.dfCVPredictions.ix[:]
        else:
            #---Hack to fix Oakland remote_api overlap on views----#
            if model.model_name == 'oakland_other_views':
                model.dfPredictions = model.dfPredictions.merge(dfTest.ix[:][['source','id']], on='id', how='left')
                model.dfPredictions = model.dfPredictions[model.dfPredictions.source != 'remote_api_created']
                del model.dfPredictions['source']
            #---End Hack-----#
            if cv == False:
                all_predictions[model.target] = pd.concat([all_predictions[model.target], model.dfPredictions])
            else:
                all_predictions[model.target] = pd.concat([all_predictions[model.target], model.dfCVPredictions])
    dfTestPredictionsTotal = all_predictions['num_views'].merge(all_predictions['num_votes'], on='id', how='left')\
            .merge(all_predictions['num_comments'], on = 'id', how='left')
    return dfTestPredictionsTotal

if __name__ == '__main__':
    sys.exit(main())