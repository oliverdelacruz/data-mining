########################################################################################################################
#Project: Data mining
#Authors: Oliver De La Cruz
#Date: 22/10/2016
#Description: Coreset algorithm for large datasets
########################################################################################################################

import numpy as np
from numpy import linalg as LA
import time
global_start = time.time()

def mapper(key, value):
       
    # Local variables
    n_clusters = 250
    n_samples = value.shape[0]   
    n_dims = value.shape[1]
    n_coresets = 2500
    alpha = np.log2(n_clusters) + 1
     
    # Initialize arrays    
    dist = np.zeros(shape=(n_samples,n_clusters))
    weights = np.ones(shape=(n_samples))/n_samples
    start_time = time.time()
    
    # Perform D2 sampling
    for idx in xrange(n_clusters):
        dist[:,idx] = map(LA.norm, value[xrange(n_samples),:] - value[np.random.choice(n_samples,1, p=weights),:])
        dist[:,idx] = dist[:,idx]**2
        min_dist = dist[:, :(idx+1)].min(axis=1)
        weights = min_dist / np.sum(min_dist)
    end_time = time.time()
    print("Time: " + str(end_time - start_time))    
    
    # Find indexes and sum distances
    idx_min = np.argmin(dist,axis=1)    
    c_phi = min_dist.sum()/n_samples

    #Construct coreset using importance sampling
    for idx in xrange(n_clusters):       
        min_dist_cluster = min_dist[np.where(idx_min == idx)]
        dist_cluster = (alpha * min_dist_cluster / c_phi)
        + (2 * alpha * min_dist_cluster.sum()/ (c_phi * min_dist_cluster.shape[0]))
        + (4 * n_samples / min_dist_cluster.shape[0])        
        if idx == 0:
            s = dist_cluster 
        else:
            s = np.hstack((s,dist_cluster))
    weights = s/s.sum()
    idx_coreset = np.random.choice(n_samples,n_coresets, p = weights)
    yield 1, np.c_[1/(weights[idx_coreset]),value[idx_coreset,:]]
    
def reducer(key, value):
    
    # Parameters
    n_restarts = 1
    n_iter = 20
    n_clusters = 200
    n_samples = value.shape[0]
    n_dims = 250
    delta = 100
    eps = 0.0001
    total_time = 0
        
    # Split information weigth/coresets
    weights = value[:,0] / n_samples
    value = value[:,1:]     
        
    # Perform weighted k-means for n_iter
    for i in range(n_restarts):

        # Initialize variables
        print "restart %i..." % (i + 1)
        idx_min = np.zeros((n_samples,))
        dist = np.zeros((n_samples,n_clusters))
        centers =  np.zeros((n_clusters,n_dims))
        prob = np.ones(shape=(n_samples))/n_samples
        bool_cond = False
        n_iterations = 0
        
        # Perform D2 sampling
        for idx in xrange(n_clusters):
            centers[idx,:] = value[np.random.choice(n_samples,1, p=prob),:]
            dist[:,idx] = map(LA.norm, value[xrange(n_samples),:] - centers[idx,:] )
            dist[:,idx] = dist[:,idx]**2
            min_dist = dist[:, :(idx+1)].min(axis=1)
            prob = min_dist / np.sum(min_dist)       
        
        # Perform Lyod's algorithm
        while bool_cond != True and delta > eps and  n_iter > n_iterations and total_time < 23.0:

            # Copy centers
            old_centers = np.copy(centers)

            # Calculate minimum euclidean distance
            start_time = time.time()
            for idx in xrange(n_clusters):
                dist[:,idx] = map(LA.norm, value[xrange(n_samples),:] - centers[idx,:])
                dist[:,idx] = dist[:,idx]**2                 
            end_time = time.time()            
            print "Time distance calculation:" + str(end_time-start_time)

            # Assign points to clusters and update centers
            min_dist = 0
            idx_min = np.argmin(dist,axis=1) 
            for idx in xrange(n_clusters):
                idx_cluster = np.array(np.where(idx_min == idx)).flatten()
                centers[idx,:] = np.average(value[idx_cluster,:],axis = 0, weights = weights[idx_cluster])
                min_dist += np.sum(dist[idx_cluster,idx] * weights[idx_cluster])

            # Compute convergence
            bool_cond = np.array_equal(centers,old_centers)
            delta = np.absolute((centers - old_centers).sum())
            n_iterations += 1
            total_time = (time.time()- global_start)/60
            print "Difference of centers sum: " + str(delta)
            print "loss Function value: " + str(min_dist)
            print "Total time: " + str(total_time)            
    yield centers
