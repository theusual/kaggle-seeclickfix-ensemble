"""
=================================================================================================
Misc code snippets used in the ipython console throughout the project for development and exploration,
but NOT directly referenced in the final program execution.
=================================================================================================
"""
__author__ = 'Bryan Gregory'
__email__ = 'bryan.gregory1@gmail.com'
__date__ = '12-24-2013'

#Internal modules
import train
import data_io

#External modules
from scipy import sparse
from sklearn.externals import joblib
import sys
import csv
import json
import numpy as np
import pandas as pd
from sklearn.datasets import dump_svmlight_file
from sklearn import (metrics, cross_validation, linear_model, ensemble, tree, preprocessing, svm, neighbors, gaussian_process, naive_bayes, neural_network, pipeline, lda)

################################################################################################
#----------------------------------------------------------------------------#
#----List of all current SKLearn learning algorithms capable of regression---#
#----------------------------------------------------------------------------#

#----For larger data sets------#
#clf = linear_model.LogisticRegression(penalty='l2', dual=True, tol=0.0001, C=1, fit_intercept=True, intercept_scaling=1.0, class_weight=None, random_state=None);clf_name='log'
#clf = linear_model.SGDRegressor(alpha=0.001, n_iter=800,shuffle=True); clf_name='SGD_001_800'
#clf = linear_model.Ridge();clf_name = 'RidgeReg'
#clf = linear_model.LinearRegression();clf_name = 'LinReg'
#clf = linear_model.ElasticNet()
#clf = linear_model.Lasso();clf_name = 'Lasso'
#clf = linear_model.LassoCV(cv=3);clf_name = 'LassoCV'
#clf = svm.SVR(kernel = 'poly',cache_size = 16000.0) #use .ravel(), kernel='rbf','linear','poly','sigmoid'
#clf = svm.NuSVR(nu=0.5, C=1.0, kernel='linear', degree=3, gamma=0.0, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=20000, verbose=False, max_iter=-1)

#----For smaller data sets------#   (Do not work or have very long training times on large sparse datasets)  Require .todense()
#clf = ensemble.RandomForestRegressor(n_estimators=50);  clfname='RFReg_50'
#clf = ensemble.ExtraTreesRegressor(n_estimators=30)  #n_jobs = -1 if running in a main() loop
#clf = ensemble.GradientBoostingRegressor(n_estimators=700, learning_rate=.1, max_depth=1, random_state=888, loss='ls');clf_name='GBM'
clf = ensemble.AdaBoostRegressor(base_estimator=tree.DecisionTreeRegressor(compute_importances=None, criterion='mse', max_depth=3,
                                                                                   max_features=None, min_density=None, min_samples_leaf=1,
                                                                                   min_samples_split=2, random_state=None, splitter='best'),
                                 n_estimators=150, learning_rate=.5, loss='linear', random_state=None)
#clf = gaussian_process.GaussianProcess(corr='cubic', theta0=1e-2, thetaL=1e-4, thetaU=1e-1, random_start=100)
#clf = neighbors.KNeighborsRegressor(100, weights='uniform', algorithm = 'auto');clf_name='KNN_200'


################################################################################################
#---Different methods of cross validation---#
#May require mtxTrn.toarray()
cv_preds = train.cross_validate(hstack([sparse.csr_matrix(dfTrn.urlid.values).transpose(),mtxTrn]),mtxTrnTarget.ravel(),
                                folds=10,SEED=42,test_size=.1,clf=clf,clf_name=clf_name,pred_fg=True)
train.cross_validate(mtxTrn,mtxTrnTarget.ravel(),folds=8,SEED=888,test_size=.1,clf=clf,clf_name=clf_name,pred_fg=False)
train.cross_validate_temporal(mtxTrn,mtxTest,mtxTrnTarget.ravel(),mtxTestTarget.ravel(),clf=clf,
                              clf_name=clf_name,pred_fg=False)
train.cross_validate_using_benchmark('global_mean',dfTrn, mtxTrn,mtxTrnTarget,folds=20)


################################################################################################
#---Calculate the degree of variance between ground truth and the mean of the CV predictions.----#
#---Returns a list of all training records with their average variance---#
train.calc_cv_preds_var(dfTrn,cv_preds)


################################################################################################
#--Use estimator for manual predictions--#
dfTest, clf = train.predict(mtxTrn,mtxTrnTarget.ravel(),mtxTest,dfTest,clf,clf_name) #may require mtxTest.toarray()
dfTest, clf = train.predict(mtxTrn.todense(),mtxTrnTarget.ravel(),mtxTest.todense(),dfTest,clf,clf_name) #may require mtxTest.toarray()

################################################################################################
#--Save feature matrices in svm format for external modeling--#
y_trn = np.asarray(dfTrn.num_votes)
y_test = np.ones(mtxTest.shape[0], dtype = int )
dump_svmlight_file(mtxTrn, y_trn, f = 'Data/Votes_trn.svm', zero_based = False )
dump_svmlight_file(mtxTest, y_test, f = 'Data/Votes_test.svm', zero_based = False )

################################################################################################
#--Save a model to joblib file--#
data_io.save_cached_object(clf,'rf_500_TextAll')

#--Load a model from joblib file--#
data_io.load_cached_object('Models/040513--rf_500_TextAll.joblib.pk1')

################################################################################################
#--Save text feature names list for later reference--#
data_io.save_text_features('Data/text_url_features.txt',tfidf_vec.get_feature_names())
