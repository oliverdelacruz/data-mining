########################################################################################################################
#Project: Data mining
#Authors: Oliver De La Cruz
#Date: 25/10/2016
#Description: Recommeder/Contextual bandit algorithm
########################################################################################################################

import numpy as np

# Lin UCB Algorithm
class LinUCB:
    def __init__(self):

        # Initialize variables
        delta = 0.201
        alpha = 1 + ((np.log(2 / delta)) / 2)**0.5
        self.alpha = alpha # Upper bound - np.sqrt(np.log(2/delta)/2)
        self.Aa = {}
        self.AaI = {}
        self.ba = {}
        self.theta = {}
        self.a_max = 0
        self.r1 = 50
        self.r0 = -6
        self.x = None
        self.xT = None
        
        # dimension of user features = d
        self.d = 6     

    def set_articles(self, art):
        
        # Initialize collection of matrix
        for key in art:
            self.Aa[key] = np.identity(self.d)
            self.ba[key] = np.zeros((self.d, 1))
            self.AaI[key] = np.identity(self.d)
            self.theta[key] = np.zeros((self.d, 1))
            
    def update(self, reward):
        
        # Update rule
        if reward == -1:
            pass
        else:
            if reward == 1:
                r = self.r1
            else:
                r = self.r0                  
            self.Aa[self.a_max] += np.dot(self.x, self.xT)
            self.ba[self.a_max] += float(r) * self.x
            self.AaI[self.a_max] = np.linalg.inv(self.Aa[self.a_max]) 
            self.theta[self.a_max] = np.dot(self.AaI[self.a_max], self.ba[self.a_max])
              
    def reccomend(self, timestamp, user_features, articles):

        # Build matrices
        xaT = np.array([user_features])
        xa = np.transpose(xaT)
        
        # Perform matrix calculations       
        AaI_tmp = np.array([self.AaI[article] for article in articles])
        theta_tmp = np.array([self.theta[article] for article in articles])
        art_max = articles[np.argmax(np.dot(xaT, theta_tmp) + self.alpha * np.sqrt(np.dot(np.dot(xaT, AaI_tmp), xa)))]

        # New user features
        self.x = xa
        self.xT = xaT
        
        # Recommented article with largest UCB
        self.a_max = art_max

        # Return output
        return self.a_max

# Setup variables
LinUCBObj = None
t = 0

def set_articles(art):

    # Global variable
    global LinUCBObj

    # Initialize class
    LinUCBObj = LinUCB()

    # Collect features from articles
    LinUCBObj.set_articles(art)

    # Return output
    pass 

def update(reward):

    # Update class
    LinUCBObj.update(reward)

    # Return output
    pass 
 
def recommend(time, user_features, choices):
    global t
    t+=1  
    return np.int64(LinUCBObj.reccomend(time, user_features, choices))
   
