import sparserbm, metrics
from sklearn import linear_model
from matplotlib.mlab import find 
import numpy as np

SparseRBM = sparserbm.SparseRBM

def apply_bound(pred, bounds=(0,1,0)):
    """
    This ensures that all views and comments >= 0 and all votes >= 1
    """
    pred = (pred >= bounds) * pred + bounds * (pred < bounds)
    return pred
    
class RidgeRMSLE(object):
    """
    A wrapper for sklearn's Ridge model that is used for regression against
    the RMSLE metric. This means that prior to training the Ridge, all values
    in the target variables are passed through a log(y + 1) transform. Also, all
    predictions from the Ridge are transformed with exp(pred) - 1. 
     
    I've also added a bounding value that can be used to ensure that all 
    predictions are bounded below by the defined bound. 
    
    The API for training, predicting, and scoring this model is similar to
    that used by sklearn's Ridge model. 
    """
    def __init__(self, alpha=1.0, bound=(0,0,0)):
        self._ridge = linear_model.Ridge()
        self.alpha = alpha
        self.bound = bound
    
    def __repr__(self):
        return 'RidgeRMSLE(alpha=%s,bound=%s)' % (self.alpha, str(self.bound))
        
    @property
    def alpha(self):
        return self._ridge.alpha
    
    @alpha.setter
    def alpha(self, val):
        self._ridge.alpha = val
    
    def fit(self, X, y):
        yt = np.log(y + 1)
        return self._ridge.fit(X, yt)
    
    def predict(self, X):
        pred = self._ridge.predict(X)
        return apply_bound(np.exp(pred) - 1, self.bound)
    
    def score(self, X, y):
        return metrics.rmsle(self.predict(X), y)
        
class ELM(object):
    """
    Implementation of an Extreme Learning Machine trained with an abitrary 
    regression algorithm
    """
    def __init__(self, hidden_size=250, transform=np.tanh, random_state=None,
                 regressor=linear_model.Ridge, **regressor_args):
        self.regressor = regressor(**regressor_args)
        self.hidden_size = hidden_size
        self.transform = transform
        self.random_state = random_state
        
    def fit(self, X, y):
        np.random.seed(self.random_state)
        self.coef_ = np.random.randn(X.shape[1], self.hidden_size)
        self.bias_ = np.random.randn(self.hidden_size)
        H = self.transform(X.dot(self.coef_) + self.bias_)
        self.regressor.fit(H, y)
        return self
    
    def predict(self, X):
        H = self.transform(X.dot(self.coef_) + self.bias_)
        return self.regressor.predict(H)

class ELMEnsemble(object):
    """
    Implementation of an Extreme Learning Machine Ensemble
    
    This trains n_estimators ELM models with distinct random states and averages
    the predictions
    """
    def __init__(self, n_estimator=10, random_state=None, **elm_args):
        self.n_estimators = n_estimators
        if random_state is None:
            self.random_state = np.random.randint(1000)
        else:
            self.random_state = random_state
        self._elms = [ELM(random_state=self.random_state+i, **elm_args) 
                      for i in range(n_estimators)]
    
    def fit(self, X, y):
        self._elms = [e.fit(X, y) for e in self._elms]
        return self
    
    def predict(self, X):
        total = self._elms[0].predict(X)
        for e in self._elms[1:]:
            total += e.predict(X)
        return total/self.n_estimators
