import numpy as np
from numpy import random
from scipy import sparse
from matplotlib.mlab import find
import time
from sys import stdout

class SparseRBM(object):
    """
    SparseRBM implements an RBM using a sparse subsampling procedure to
    speed up runtime. The idea is to perform gibbs sampling on a subset of
    the visible layer instead of the full visible layer. This is achieved by
    passing along a list of indexes for the active features in each gibbs step. 
    Gibbs is run on the active features while all other features are assumed to 
    have an activation probability of 0.
    
    This current implementation only considered features that are active in at 
    least one data point in the mini batch. Other selection procedures can be
    used by implementing a different 'make_batches' procedure in the source code
    
    Training is performed using stochastic gradient descent (SGD) with momentum
    using fast persistent contrastive divergence (FPCD-n)
    
    Documentation on RBM: http://www.cs.toronto.edu/~hinton/absps/guideTR.pdf
    Documentation on FPCD: http://www.cs.toronto.edu/~tijmen/fpcd/fpcd.pdf
    
    Parameters:
        visible_size: The size of the visible layer
        hidden_size: The size of hidden layer
        epochs: The number of training iterations over the dataset (default 10)
        batch_size: The size of each individual mini-batch (default 100)
        n: The number of gibbs steps in the negative phase of training (default 1)
        learn_rate: The learning rate of SGD (default 0.1)
        momentum: The momentum of gradient updates in SGD (default 0.9)
        fw_decay: The decay rate of the fast-weights in FPCD 
                  (the default of 0.98 gives good emperical results)
        l2: The magnitude of L2 regularization (default 0.0)
        verbose: Display costs and runtime during training (default False)
    
    Set attributes:
        W: The weights of the RBM
        vbias: The visible bias term
        hbias: The hidden bias term
        fW: The fast-weights used for FPCD
        total_epochs: The total number of epochs this RBM has been trained
        cost_hist: A list of cost histories of each mini-batch during training
    """
    def __init__(self, visible_size, hidden_size, epochs=10, batch_size=100, 
                 n=1, learn_rate=0.1, momentum=0.9, fw_decay=0.98, l2=0.0, 
                 verbose=False):
        self.visible_size = visible_size
        self.hidden_size = hidden_size
        self.hbias = np.zeros(hidden_size)
        self.vbias = np.zeros(visible_size)
        self.W = random.randn(hidden_size, visible_size) / \
                 np.sqrt(hidden_size + visible_size)
        self.batch_size = batch_size
        self.epochs = epochs
        self.learn_rate = learn_rate
        self.n = n
        self.l2 = l2
        self.momentum = momentum
        self.verbose = verbose
        self.fw_decay = fw_decay
        self.fW = np.zeros(self.W.shape)
        self.flr = learn_rate * np.exp(1) # fast learn rate heuristic
        self.p = np.zeros((self.batch_size, self.hidden_size))
        self._prevgrad = {'W': np.zeros(self.W.shape), 
                          'hbias': np.zeros(hidden_size), 
                          'vbias': np.zeros(visible_size)}
        self.total_epochs = 0
        self.cost_hist = []
    
    def __repr__(self):
        return ('SparseRBM(visible_size=%i, hidden_size=%i, epochs=%i, '
                'batch_size=%i, learn_rate=%.4g, momentum=%.4g, l2=%.4g, '
                'fw_decay=%.4g, verbose=%s)') %(self.visible_size, self.hidden_size,
                 self.epochs, self.batch_size, self.learn_rate, self.momentum, 
                 self.l2, self.fw_decay, self.verbose)
                 
    def propup(self, vis, subsample_ids=None, fw=False):
        """
        Compute and sample P(h | v) 
        
        Note: Calling this function without subsample_ids can be used to perform
              a transformation of the full dataset to be used in a ML pipeline
                      
        Args:
            vis: The state of the visible layer - shape (m, len(subsample_ids))
            subsample_ids: The feature indices used in this subsample
            fw: Boolean controlling if fast weights should be used
        
        Returns:
            a 3-tuple: (sample, probabilities, linear ouputs before nonlinearity)
        """
        W = self.fW + self.W if fw else self.W
        if subsample_ids is not None:
            W = W[:, subsample_ids]
        pre_non_lin = vis.dot(W.T) + self.hbias
        non_lin = sigmoid(pre_non_lin)
        sample = sample_bernoulli(non_lin)
        return (sample, non_lin, pre_non_lin)
    
    def propdown(self, hid, subsample_ids=None, fw=False):
        """
        Compute and sample P(v | h) 
        
        Note: Calling this function without subsample_ids with a large 
              visible_size will be extremely slow and may potentially cause
              memory issues resulting in massive slowdowns of your computer.
              You've been warned! 
        
        Args:
            hid: The state of the hidden layer - shape: (m, hidden_size)
            subsample_ids: The visible indices we want from this subsample
            fw: Boolean controlling if fast weights should be used
        
        Returns:
            a 3-tuple: (sample, probabilities, linear ouputs before nonlinearity)
        """
        W = self.fW + self.W if fw else self.W
        vbias = self.vbias
        if subsample_ids is not None:
            W = W[:, subsample_ids]
            vbias = vbias[subsample_ids]
        pre_non_lin = hid.dot(W) + vbias
        non_lin = sigmoid(pre_non_lin)
        sample = sample_bernoulli(non_lin)
        return (sample, non_lin, pre_non_lin)
    
    def gibbs_hvh(self, h, meanfield=False, **args):
        """
        Performs one step of gibbs sampling given the hidden state
        
        Args:
            h: The hidden state
            meanfield: Boolean controlling if we want to use the mean field values
                       during gibbs instead of samples
            **args: arguments to pass to propup/propdown procedures
            
        Returns:
            a 2-tuple of 3-tuples (visible samples, hidden samples)
        """
        v_samples = self.propdown(h, **args)
        v = v_samples[1] if meanfield else v_samples[0]
        h_samples = self.propup(v, **args)
        return v_samples, h_samples
    
    def gibbs_vhv(self, v, meanfield=False, **args):
        """
        Performs one step of gibbs sampling given the visible state
        
        Args:
            v: The visible state
            meanfield: Boolean controlling if we want to use the mean field values
                       during gibbs instead of samples
            **args: arguments to pass to propup/propdown procedures
            
        Returns:
            a 2-tuple of 3-tuples (visible samples, hidden samples)
        """
        h_samples = self.propup(v, **args)
        h = h_samples[1] if meanfield else h_samples[-1]
        v_samples = self.propdown(h, **args)
        return v_samples, h_samples
    
    def cost(self, v, subsample_ids=None):
        """
        Compute the 'cost' and gradient using FPCD.
        
        NOTE: The 'cost' is not an actual cost metric, it is only the 
              approximate reconstruction error of the visible sample. What RBMs
              are actually minimizing is an energy function
        
        Args:
            v: The visible state
            subsample_ids: The visible indices we want to consider when computing
                           the reconstruction error and gradient
        
        Returns:
            cost: The reconstruction error
            grad: A dict containing gradient approximations for W, vbias, hbias
        """
        num_points = v.shape[0]
        # positive phase
        pos_h_samples = self.propup(v, subsample_ids)
        # negative phase
        nh0 = self.p[:num_points]
        for i in xrange(self.n):
            neg_v_samples, neg_h_samples = self.gibbs_hvh(nh0, 
                                                    subsample_ids=subsample_ids, 
                                                    fw=True)
            nh0 = neg_h_samples[0]
        # compute gradients
        grad = self._grad(v, pos_h_samples, neg_v_samples, neg_h_samples)
        self.p[:num_points] = nh0
        # compute reconstruction error
        reconstruction = self.propdown(pos_h_samples[0], subsample_ids)[1]
        cost = np.abs(v - reconstruction).sum(1).mean(0)
        return cost, grad
    
    def _grad(self, pv0, pos_h, neg_v, neg_h):
        """
        Helper to compute the gradient approximation
        
        Args:
            pv0: visible layer state from the positive phase
            pos_h: hidden layer state from the postive phase
            neg_v: visible layer state from the negative phase
            neg_h: hidden layer state from the negative phase
        
        Returns:
            The gradient dict required in the cost function
        """
        grad = {}
        num_points = pv0.shape[0]
        E_v = neg_v[1]
        E_h = neg_h[1]
        E_hgv = pos_h[1]
        E_vh = E_h.T.dot(E_v)
        E_vhgv = E_hgv.T.dot(pv0)
        grad['W'] = (E_vhgv - E_vh) / num_points
        grad['vbias'] = (pv0 - E_v).mean(0)
        grad['hbias'] = (E_hgv - E_h).mean(0)
        return grad
    
    def update(self, grad, subsample_ids=None):
        """
        Update the RBM parameters W, vbias, hbias, fW using momentum
        
        Args:
            grad: The gradient dict returned from the cost function
            subsample_ids: The subsample indices used when generating the gradient
        
        Returns:
            self
        """
        prev_grad = self._prevgrad
        dW0 = grad['W']
        dv0 = grad['vbias']
        dh0 = grad['hbias']
        if subsample_ids is not None:
            dv0 = np.zeros(self.vbias.shape)
            dW0 = np.zeros(self.W.shape)
            dv0[subsample_ids] = grad['vbias']
            dW0[:, subsample_ids] = grad['W']
        dW = self.momentum * prev_grad['W'] + \
             self.learn_rate * (dW0 - self.l2 * self.W)
        dh = self.momentum * prev_grad['hbias'] + self.learn_rate * dh0
        dv = self.momentum * prev_grad['vbias'] + self.learn_rate * dv0
        self.W += dW
        self.hbias += dh
        self.vbias += dv
        self.fW = self.fw_decay * self.fW + self.flr * dW0 # Fast weight update for PCD
        self._prevgrad['W'] = dW
        self._prevgrad['hbias'] = dh
        self._prevgrad['vbias'] = dv
        return self
    
    def transform(self, data):
        """
        Perform a transformation of the data to activation probabilities of the
        hidden layer
        
        Args:
            data: the data to be transformed interpreted as the visible layer
        
        Returns:
            The activation probabilities p(h | v)
        """
        return self.propup(data)[1]
        
    def fit(self, data):
        """
        Trains the RBM using stochastic gradient descent for self.epochs 
        iterations over the dataset
        
        Note: contrary to idioms from Scikit-Learn, calling fit will not 
              reinitialize the weights of the model. Training will continue 
              given the weights and biases the RBM has configured
              
        Args:
            data - the data to be interpreted as the visible states of the RBM
        
        Returns:
            self
        """
        n, m = data.shape
        num_batches = n / self.batch_size
        e = 0
        
        if self.verbose: 
            start_time = time.clock()
        
        while e < self.epochs:
            e += 1
            batches = make_batches(data, self.batch_size)
            for i, (batch, subsample_ids) in enumerate(batches):
                cost, grad = self.cost(batch, subsample_ids)
                self = self.update(grad, subsample_ids)
                self.cost_hist.append(cost)
                if self.verbose:
                    print 'Batch %i - Cost %0.6f\r'%(i+1, cost),
                    stdout.flush()
            if self.verbose:
                print 'Training Epoch %i'%(self.total_epochs), 
                print 'Average Cost: %0.6f\t\t'%np.mean(self.cost_hist[-num_batches:])
                stdout.flush()
            self.total_epochs += 1
            
        if self.verbose: 
            end_time = time.clock()
            print 'Runtime %0.2fs'%(end_time-start_time)
            
        return self

def make_batches(data, batch_size=100):
    """
    Split the data into minibatches of size batch_size
    
    This procedure generates subsamples ids for batches by only considering 
    features that are active in the minibatch
    
    Args:
        data - the data to be split into minibatches (must be rank 2)
        batch_size - the size of the minibatches 
        
    Returns:
        batches - a list: [(batch, subsample_ids) for batch in minibatchs]
    """
    n = data.shape[0]
    perm = random.permutation(range(n))
    i = 0
    batches = []
    while i < n:
        batch = perm[i:i+batch_size]
        i += batch_size
        batches.append(data[batch])
    try:
        ids = [find((b.sum(0) != 0).A.flatten()) for b in batches]
    except AttributeError:
        ids = [find((b.sum(0) != 0).flatten()) for b in batches]
    batches = [(b[:,i].toarray(), i) for b,i in zip(batches, ids)]
    return batches
    
def sigmoid(X):
    """Compute sigmoid function"""
    return 1 / (1 + np.exp(-X))

def sample_bernoulli(X):
    """
    All values of X must be probabilities of independent events occuring according
    to a binomial distribution
    
    Returns an indicator array:
        output[i,j] = 1 iif X[i,j] >= uniform(0, 1)
    """
    return (X >= random.uniform(size=X.shape)).astype('b')
