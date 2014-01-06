import pandas as pd
import numpy as np

import logging

import config, utils, datasets, model_types, predict, cross_validation

logger = logging.getLogger(__name__)

def make_model(name):
    cfgs = config.model_configs(name)
    try:
        model_class = getattr(model_types, cfgs['model'])
    except AttributeError:
        raise AttributeError('Unable to find model \
                               %s in model_types.py' % cfgs['model'])
    logger.info('Creating model %s' % name)
    model = model_class(**cfgs['args'])
    return model

def train_model(name):
    try:
        model = utils.load_from_cache(name)
        logger.info('Loading model %s from cache' % name)
    except IOError:
        cfgs = config.model_configs(name)
        model = make_model(name)
        data = get_model_data(name)
        logger.info('Training model %s' % name)
        if "target" in cfgs:
            (train_data, target), test_data = data
            model.fit(train_data, target)
        else:
            model.fit(data)
        utils.save_to_cache(model, name)
    return model

def get_model_data(name):
    cfgs = config.model_configs(name)
    data = datasets.load_dataset(cfgs['dataset'])
    if 'target' in cfgs:
        target = datasets.load_dataset(cfgs['target'])
        n_train = target.shape[0]
        train_data, test_data = cross_validation.train_test_split(data, n_train)
        data = ((train_data, target), test_data)
    return data 

def predict_model(name, data, model=None):
    if model is None:
        model = train_model(name)
       
    try:
        pred = model.predict(data)
    except AttributeError:
        raise AttributeError("Model %s does not implement a predict function" % name)
    
    cfgs = config.model_configs(name)     
    if 'postprocess_pred' in cfgs:
        postprocess_pred = getattr(predict, cfgs['postprocess_pred']['name'])
        pred = postprocess_pred(pred, **cfgs['postprocess_pred'].get('args', {})) 
    
    return pred
    
def test_model(name):
    (train_data, target), test_data = get_model_data(name)
    return predict_model(name, test_data)
    
def validate_model(name, return_data=False):
    cfgs = config.model_configs(name)
    (train_data, target), test_data = get_model_data(name)
    
    if 'validator' not in cfgs:
        validator = cross_validation.train_test_split
        args = { 'n_train': 0.8 }
    else:
        validator = getattr(cross_validation, cfgs['validator']['name'])
        args = cfgs['validator'].get('args', {})
        
    X_train, X_cv = validator(train_data, **args)
    y_train, y_cv = validator(target, **args)
    model = make_model(name)
    model.fit(X_train, y_train)
    score = model.score(X_cv, y_cv)
    return (score, (model.predict(X_cv), y_cv)) if return_data else score
    
