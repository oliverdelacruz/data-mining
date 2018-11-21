########################################################################################################################
#Project: Data mining
#Authors: Oliver De La Cruz
#Date: 22/10/2016
#Description: Online Adam optimization algorithm with Polynomial Function
########################################################################################################################

import numpy as np
from itertools import chain, combinations

def transform(X):
    # Make sure this function works for both 1D and 2D NumPy arrays.

    # Local variables
    n_samples = X.shape[0]
    n_features = X.shape[1]
    n_degrees = 2

    # Create combinations of the features
    def combinations_with_replacement(iterable, r):
        # combinations_with_replacement('ABC', 2) --> AA AB AC BB BC CC
        pool = tuple(iterable)
        n = len(pool)
        if not n and r:
            return
        indices = [0] * r
        yield tuple(pool[i] for i in indices)
        while True:
            for i in reversed(range(r)):
                if indices[i] != n - 1:
                    break
            else:
                return
            indices[i:] = [indices[i] + 1] * (r - i)
            yield tuple(pool[i] for i in indices)

    # Performs the combinations for a given degree
    def combinations(n_features, degree):        
        start = 0
        return chain.from_iterable(combinations_with_replacement(range(n_features), i)
                                   for i in range(start, degree + 1))

    # Call functions
    combinations = combinations(n_features,n_degrees)

    # Get dimensions of the new array
    # 2 - degrees = 80601
    # 3 - degrees = 10827401
    n_output_features =  80601 

    # Initialize array
    XP = np.empty((n_samples, n_output_features))
    
    # Generate the matrix
    for i, c in enumerate(combinations):
        XP[:, i] = X[:, c].prod(1)        
    
    return XP * 1000


def mapper(key, value):

    # Local variables 7
    T = 1
    param_labmda = 0
    
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
        for i in xrange(0, num_iter):
            if (y[i] * np.dot(w, x[i])) < 1:
                gradient = - y[i] * x[i]                
            else:
                gradient = np.zeros((num_dims,))
            m = beta1 * m + (1 - beta1) * gradient
            v = beta2 * v + (1 - beta2) * (gradient**2)
            mt = m / (1-(beta1**(i+1)))
            vt = v/ (1-(beta2**(i+1)))
            w += - alpha * mt / (np.sqrt(vt) + epsilon)
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
    train_features, targets = random_sample(train_features, targets, T)

    # Transform features
    train_features = transform(train_features)
    
    # Calculate and yield weight
    w = adam_svm(train_features, targets,param_labmda)
    yield 1, w


def reducer(key, values):
    yield sum(values)/len(values)
