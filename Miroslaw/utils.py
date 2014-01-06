import cPickle as pickle
import datasets, config
import numpy as np
import pandas as pd
from sklearn.externals import joblib

def load_from_cache(filename):
    filename = '%s/%s.pkl' % (config.CACHEDIR, filename)
    if config.CACHETYPE == 'joblib':
        obj = joblib.load(filename)
    elif config.CACHETYPE == 'pickle':
        f = open(filename)
        obj = pickle.load(f)
        f.close()
    else:
        raise ValueError('Unkown CACHETYPE %s, use only pickle or joblib' % config.CACHETYPE)
    return obj

def save_to_cache(obj, filename):
    filename = '%s/%s.pkl' % (config.CACHEDIR, filename)
    if config.CACHETYPE == 'joblib':
        joblib.dump(obj, filename, compress=9)
    elif config.CACHETYPE == 'pickle':
        f = open('cache/%s.pkl' % filename, 'w')
        pickle.dump(obj, f, 2)
        f.close()
    else:
        raise ValueError('Unkown CACHETYPE %s, use only pickle or joblib' % config.CACHETYPE)

def create_submission(filename, pred, ids=None):
    data = ['id,num_views,num_votes,num_comments']
    pred = pred.astype('S100')
    
    if ids is None:
        ids = datasets.load_dataset('TestIDS')
    
    for id, p in zip(ids, pred):
        data.append('%i,' % (id) + ','.join(p))
    data = '\n'.join(data)
    f = open('%s/%s' %(config.SUBMITDIR, filename), 'w')
    f.write(data)
    f.close()

def make_vw(data, targets, filename):
    """
    Helper method to create a vowpal wabbit dataset from data and targets and
    save it to filename
    """
    s = []
    for yi, xi in zip(targets, data):
        xis = ' '.join(['f%i:%f' % (f, xi[0,f]) for f in xi.nonzero()[1]])
        s.append('%f | %s' %(yi, xis))
    f = open(filename, 'w')
    f.write('\n'.join(s))
    f.close()
    
def greedy_feature_selection(model, features, j):
    selected_features = set()
    score_hist = []
    ycv = exp(y_cv) - 1
    while len(selected_features) < len(features):
        scores = []
        for i in range(len(features)):
            if i not in selected_features:
                feats = list(selected_features) + [i]
                if len(feats) == 1:
                    ttfs = features[i]
                else:
                    ttfs = data_transforms.drop_disjoint(sparse.hstack((
                            features[feats])).tocsr(), targets)
                X_train_pre = ttfs[:n_train]
                X_train = X_train_pre[:int(n_train*0.8)]
                X_cv = X_train_pre[int(n_train*0.8):]
                model.fit(X_train[-keep:], y_train[-keep:])
                cv = exp(ridge.predict(X_cv)) - 1
                scores.append((rmsle(postprocess_pred(cv)[:,j], ycv[:,j]), feats, i))
                print scores[-1]
        selected_features.add(min(scores)[2])
        score_hist.append(min(scores))
    return score_hist
