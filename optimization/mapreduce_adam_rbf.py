########################################################################################################################
#Project: Data mining
#Authors: Oliver De La Cruz
#Date: 22/10/2016
#Description: Online Adam optimization algorithm
########################################################################################################################

import numpy as np
import math

# Global variables
np.random.seed(7)    
n_samples = 2500
n_features = 400
mean = np.zeros((n_features))    
gamma = 193
cov = np.identity(n_features) * gamma

# Generate random numbers   
w = np.transpose(np.random.multivariate_normal(mean,cov, n_samples)) 
b = np.random.uniform(0, 2 * math.pi, n_samples)

def transform(X):
    # Make sure this function works for both 1D and 2D NumPy arrays.    
       
    # Calculate the matriz Z
    global w
    global b
    X = math.sqrt(float(2)/n_samples) * np.cos(np.dot(X,w)+ b)    
       
    return X * 10000

def mapper(key, value):

    # Local variables
    T = 20
    param_labmda = 0.0001
    
    # Helper functions
    def random_sample(x, y,T):
        num_samples = x.shape[0]
        idx = np.random.randint(num_samples, size= T * num_samples)        
        return x[idx, :], y[idx]
    
    # Define function to perform the calibration of the svm
    def adam_svm(x, y, l):
        alpha = 0.001
        beta1 = 0.9
        beta2 = 0.999
        epsilon =  1e-10
        num_iter = x.shape[0]
        num_dims = x.shape[1]
        w = np.zeros((num_dims,))        
        m = np.zeros((num_dims,))
        v = np.zeros((num_dims,))
        alphat =  np.zeros((num_dims,))
        for i in xrange(0, num_iter):
            if (y[i] * np.dot(w, x[i])) < 1:
                gradient = - y[i] * x[i]                
            else:
                gradient = np.zeros((num_dims,))
            m = beta1 * m + (1 - beta1) * gradient
            v = beta2 * v + (1 - beta2) * (gradient**2)
            mt = m / (1-(beta1**(i+1)))
            vt = v/ (1-(beta2**(i+1)))
            alphat = alpha * math.sqrt(1- (beta2**(i+1))) / (1 - (beta1**(i+1)))
            w += - alphat * mt / (np.sqrt(vt) + epsilon)
        return w
    
    # Generate train features and targets
    train_features = []
    targets = []
    for image in value:
        data = map(float, image.split())
        targets.append(data[0])
        train_features.append(data[1:])

    # Create the arrays used for calibration of the svm
    train_features = np.array(train_features)
    targets = np.array(targets)

    # Transform features
    train_features = transform(train_features)

    # Sample data points
    train_features, targets = random_sample(train_features, targets, T)   
 
    # Calculate and yield weight
    w = adam_svm(train_features, targets,param_labmda)

    yield 1, w


def reducer(key, values):
    yield sum(values)/len(values)
