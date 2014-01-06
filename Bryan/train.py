"""
Functions for training estimators, performing cross validation, and making predictions
"""
__author__ = 'Bryan Gregory'
__email__ = 'bryan.gregory1@gmail.com'
__date__ = '11-19-2013'

#Internal modules
import utils
#Start logger to record all info, warnings, and errors to Logs/logfile.log
log = utils.start_logging(__name__)
import ml_metrics

#External modules
import time
from datetime import datetime
from sklearn import (metrics, cross_validation, linear_model, preprocessing)
from sklearn.externals import joblib
import numpy as np
from scipy import sparse
from scipy.sparse import coo_matrix, hstack, vstack

#-----Run Cross Validation Steps-----#
def cross_validate(model, settings, dfTrn_Segment, dfTest_Segment):
    #Combine the train and test feature matrices and create targets
    mtxTrn, mtxTest, mtxTrnTarget, mtxTestTarget = combine_features(model, dfTrn_Segment, dfTest_Segment)
    #Run CV
    if settings['cv_method'] in ['march','april','list_split']:
         cv_preds = cross_validate_temporal(mtxTrn,mtxTest,mtxTrnTarget.ravel(),mtxTestTarget.ravel(),model)
    if settings['cv_method'] in ['kfold']:
         cv_preds = cross_validate_kfold(mtxTrn,mtxTest,mtxTrnTarget.ravel(),mtxTestTarget.ravel(),model)
    dfTest_Segment[model.target] = [x for x in cv_preds]

#-----Combine the train and test feature matrices and create targets-----#
def combine_features(model, dfTrn, dfTest):
    #Create targets
    mtxTrnTarget = dfTrn.ix[:,[model.target]].as_matrix()
    mtxTestTarget = dfTest.ix[:,[model.target]].as_matrix()
    #Combine train and test features
    for feature in model.features:
        if 'mtxTrn' in locals():
            #if not the first feature in the list, then add the current feature
            mtxTrn = hstack([mtxTrn, model.features[feature][0]])
            mtxTest = hstack([mtxTest, model.features[feature][1]])
        else:
            #if the first feature in the list, then create the matrices
            mtxTrn = model.features[feature][0]
            mtxTest = model.features[feature][1]
    return mtxTrn, mtxTest, mtxTrnTarget, mtxTestTarget

#---Traditional K-Fold Cross Validation----#
def cross_validate_kfold(mtxTrn,mtxTarget,model,folds=5,SEED=42,test_size=.15,pred_fg='false'):
    fold_scores = []
    SEED = SEED *  time.localtime().tm_sec
    start_time = datetime.now()
    log.info('K-Fold CV started at: %s' % (datetime.now().strftime('%m-%d-%y %H:%M')))
    utils.line_break()
    #If predictions are wanted, initialize the dict so that its length will match all records in the training set,
    #even if not all records are predicted during the CV (randomness is a bitch)
    if pred_fg == 'true':
        cv_preds = {key[0]:[] for key in mtxTrn.getcol(0).toarray()}
    for i in range(folds):
        ##For each fold, create a test set (test_cv) by randomly holding out test_size% of the data as CV set
        train_cv, test_cv, y_target, y_true = \
           cross_validation.train_test_split(mtxTrn, mtxTarget, test_size=test_size, random_state=i*SEED+1)
        #If target variable has been transformed, transform y_true back to normal state for comparison to predictions
        y_true = [np.exp(x)-1 for x in y_true]
        #if predictions are wanted, parse off the first row from train and test cv sets. First row contains ID
        if pred_fg == 'true':
            #TODO: create dense matrix copies for the clf's that only use dense matrices
            train_cv = sparse.csr_matrix(train_cv)[:,1:]
            test_cv2 = sparse.csr_matrix(test_cv)[:,1:]
            test_cv = sparse.csr_matrix(test_cv)[:,1:]
        #----------Hyperparameter optimization------#
        try:
            model.estimator.fit(train_cv, y_target)
            preds = model.estimator.predict(test_cv)
        except TypeError:
            model.estimator.fit(train_cv.todense(), y_target)
            preds = model.estimator.predict(test_cv.todense())
        preds = model.estimator.predict(test_cv)
        #----------Post processing rules----------#
        #If target variable has been transformed, transform predictions back to original state
        preds = [np.exp(x)-1 for x in preds]
        #Apply scalar
        if model.postprocess_scalar != 1:
            preds = [x*model.postprocess_scalar for x in preds]
        #set <0 predictions to 0 if views or comments, set <1 predictions to 1 if votes
        if model.target == 'num_votes':
            preds = [1 if x < 1 else x for x in preds]
        else:
            preds = [0 if x < 0 else x for x in preds]
        ##For each fold, score the prediction by measuring the error using the chosen error metric
        score = ml_metrics.rmsle(y_true, preds)
        fold_scores += [score]
        log.info('RMLSE (fold %d/%d): %f' % (i + 1, folds, score))
        ##IF we want to record predictions, then for each fold add the predictions to the cv_preds dict for later output
        if pred_fg == 'true':
            for i in range(0,test_cv2.shape[0]):
                if test_cv2.getcol(0).toarray()[i][0] in cv_preds.keys():
                    cv_preds[test_cv2.getcol(0).toarray()[i][0]] += [preds[i]]
                else:
                    cv_preds[test_cv2.getcol(0).toarray()[i][0]] = [preds[i]]
    ##Now that folds are complete, calculate and print the results
    finish_time = datetime.now()
    log.info('Prediction metrics: mean=%f, std dev=%f, min/max= %f/%f' %
            (np.mean(fold_scores)), (np.max(fold_scores),np.std(fold_scores),np.min(fold_scores),np.max(fold_scores)))
    utils.line_break()
    log.info('K-Fold CV completed at: %s.  Total runtime: %s' % (datetime.now().strftime('%m-%d-%y %H:%M'),
                                                              str(finish_time-start_time)))
    utils.line_break()
    if pred_fg == 'true':
        return cv_preds

#---Temporal cross validation---#
def cross_validate_temporal(mtxTrn,mtxTest,mtxTrnTarget,mtxTestTarget,model):
    start_time = datetime.now()
    log.info('Temporal CV started at: %s' % (datetime.now().strftime('%m-%d-%y %H:%M')))
    utils.line_break()
    train_cv = mtxTrn
    test_cv = mtxTest
    y_target = mtxTrnTarget
    y_true = mtxTestTarget
    #If target variable has been transformed, transform y_true back to normal state for comparison to predictions
    y_true = [np.exp(x)-1 for x in y_true]
    #--------Hyperparameter optimization---------#
    #Make predictions
    try:
        model.estimator.fit(train_cv, y_target)
        preds = model.estimator.predict(test_cv)
    except TypeError:
        model.estimator.fit(train_cv.todense(), y_target)
        preds = model.estimator.predict(test_cv.todense())
    #----------Post processing rules----------#
    #If target variable has been transformed, transform predictions back to original state
    preds = [np.exp(x)-1 for x in preds]
    #Apply scalar
    if model.postprocess_scalar != 1:
        preds = [x*model.postprocess_scalar for x in preds]
    #set <0 predictions to 0 if views or comments, set <1 predictions to 1 if votes
    if model.target == 'num_votes':
        preds = [1 if x < 1 else x for x in preds]
    else:
        preds = [0 if x < 0 else x for x in preds]
    ##score the prediction by measuring the error using the chosen error metric
    score = ml_metrics.rmsle(y_true, preds)
    finish_time = datetime.now()
    log.info('Error Measure:' , score)
    log.info('Prediction metrics: mean=%f, std dev=%f, min/max= %f/%f' %
               (np.mean(preds)), (np.max(preds),np.std(preds),np.min(preds),np.max(preds)))
    utils.line_break()
    log.info('Temporal CV completed at: %s.  Total runtime: %s' \
          % (datetime.now().strftime('%m-%d-%y %H:%M'),str(finish_time-start_time)))
    utils.line_break()
    return preds

def cross_validate_using_benchmark(benchmark_name, dfTrn, mtxTrn,mtxTarget,model,folds=5,SEED=42,test_size=.15):
    fold_scores = []
    SEED = SEED *  time.localtime().tm_sec
    start_time = datetime.now()
    log.info('Benchmark CV started at: %s' % (datetime.now().strftime('%m-%d-%y %H:%M')))
    utils.line_break()
    for i in range(folds):
        #For each fold, create a test set (test_holdout) by randomly holding out X% of the data as CV set, where X is test_size (default .15)
        train_cv, test_cv, y_target, y_true = cross_validation.train_test_split(mtxTrn, mtxTarget, test_size=test_size, random_state=SEED*i+10)
        #If target variable has been transformed, transform y_true back to normal state for comparison to predictions
        y_true = [np.exp(x)-1 for x in y_true]
        #Calc benchmarks and use them to make a prediction
        benchmark_preds = 0
        if benchmark_name =='global_mean':
            benchmark_preds = [13.899 for x in test_cv]
        if benchmark_name =='all_ones':
            #find user avg stars mean
            benchmark_preds = [1 for x in test_cv]
        if benchmark_name =='9999':
            #find user avg stars mean
            benchmark_preds = [9999 for x in test_cv]
        log.info('Using benchmark %s:' % (benchmark_name))
        #For this CV fold, measure the error
        score = ml_metrics.rmsle(y_true, benchmark_preds)
        #print score
        fold_scores += [score]
        log.info('RMSLE (fold %d/%d): %f' % (i + 1, folds, score))

    ##Now that folds are complete, calculate and print the results
    finish_time = datetime.now()
    log.info('Prediction metrics: mean=%f, std dev=%f, min/max= %f/%f' %
            (np.mean(fold_scores)), (np.max(fold_scores),np.std(fold_scores),np.min(fold_scores),np.max(fold_scores)))
    utils.line_break()
    log.info('CV completed at: %s.  Total runtime: %s' % (datetime.now().strftime('%m-%d-%y %H:%M'),
                                                       str(finish_time-start_time)))
    utils.line_break()

def predict(mtxTrn,mtxTarget,mtxTest,dfTest,model):
    start_time = datetime.now()
    log.info('Predictions started at: %s' % (datetime.now().strftime('%m-%d-%y %H:%M')))
    try:
        #make predictions on test data and store them in the test data frame
        model.estimator.fit(mtxTrn, mtxTarget)
        dfTest[model.target] = [x for x in model.estimator.predict(mtxTest)]
    except TypeError:
        model.estimator.fit(mtxTrn.todense(), mtxTarget)
        dfTest[model.target] = [x for x in model.estimator.predict(mtxTest.todense())]
    #---------Post processing rules--------------#
    #If target variable has been transformed, transform predictions back to original state
    dfTest[model.target] = [np.exp(x) - 1 for x in dfTest[model.target]]
    #Apply scalar
    if model.postprocess_scalar != 1:
        dfTest[model.target] = [x*model.postprocess_scalar for x in dfTest[model.target]]
    #set <0 predictions to 0 if views or comments, set <1 predictions to 1 if votes
    if model.target == 'num_votes':
        dfTest[model.target] = [1 if x < 1 else x for x in dfTest[model.target]]
    else:
        dfTest[model.target] = [0 if x < 0 else x for x in dfTest[model.target]]
    #print 'Coefs for',model.estimator_name,model.estimator.coef_
    finish_time = datetime.now()
    log.info('Prediction metrics: mean=%f, std dev=%f, min/max= %f/%f' %
        (np.mean(dfTest[model.target]), np.std(dfTest[model.target]),np.min(dfTest[model.target]),
         np.max(dfTest[model.target])))
    log.info('Predictions completed at: %s.  Total runtime: %s' % (datetime.now().strftime('%m-%d-%y %H:%M'),
                                                                str(finish_time-start_time)))
    return dfTest

#---Calculate the variance between ground truth and the mean of the CV predictions.----#
#---Adds the average cv variance to the training dataframe for later analysis--------------------#
def calc_cv_preds_var(df, cv_preds):
    df['cv_preds_var'] = ''
    df['cv_preds_mean'] = ''
    for key in cv_preds.keys():
        df['cv_preds_var'][df.urlid == key] = abs(df[df.urlid == key].label.values[0] - np.mean(cv_preds[key]))
        df['cv_preds_mean'][df.urlid == key] = np.mean(cv_preds[key])




