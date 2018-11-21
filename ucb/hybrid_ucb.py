########################################################################################################################
#Project: Data mining
#Authors: Oliver De La Cruz
#Date: 28/10/2016
#Description: Recommeder/Contextual bandit algorithm (Hybrid)
########################################################################################################################

import numpy as np

# Lin UCB Algorithm
class HybridUCB:
    def __init__(self):

        # Initialize variables        
        self.alpha = 3.05
        self.r1 = 0.9
        self.r0 = -20 
        self.article_features = {}
        self.a_max = 0
                
        # dimension of user features = d
        self.d = 6
        
        # dimension of the article features = k
        self.k = self.d*self.d

        # Initialize matrices
        self.A0 = np.identity(self.k)
        self.A0I = np.identity(self.k)
        self.b0 = np.zeros((self.k, 1))
        self.beta = np.zeros((self.k, 1))  
                
        # Initialize other dicts
        self.AaIba = {}
        self.AaIBa = {}
        self.BaTAaI = {}
        self.theta = {}
        self.index = {}            

        # Set up user/article features
        self.z = None
        self.zT = None
        self.xaT = None
        self.xa = None           

    def set_articles(self, art):
        
        # Initialize collection of matrix
        i = 0
        art_len = len(art)
        self.article_features = np.zeros((art_len, 1, self.d))
        self.Aa = np.zeros((art_len, self.d, self.d))
        self.AaI = np.zeros((art_len, self.d, self.d))
        self.Ba = np.zeros((art_len, self.d, self.k))
        self.BaT = np.zeros((art_len, self.k, self.d))
        self.ba = np.zeros((art_len, self.d, 1))
        self.AaIba = np.zeros((art_len, self.d, 1))
        self.AaIBa = np.zeros((art_len, self.d, self.k))
        self.BaTAaI = np.zeros((art_len, self.k, self.d))
        self.theta = np.zeros((art_len, self.d, 1))
        for key in art:
            self.index[key] = i
            self.article_features[i] = np.array(art[key])
            self.Aa[i] = np.identity(self.d)
            self.AaI[i] = np.identity(self.d)
            self.Ba[i] = np.zeros((self.d, self.k))
            self.BaT[i] = np.zeros((self.k, self.d))
            self.ba[i] = np.zeros((self.d, 1))
            self.AaIba[i] = np.zeros((self.d, 1))
            self.AaIBa[i] = np.zeros((self.d, self.k))
            self.BaTAaI[i] = np.zeros((self.k, self.d))
            self.theta[i] = np.zeros((self.d, 1))
            i += 1
            
    def update(self, reward):
        
        # Update rule
        if reward == -1:
            pass
        elif reward == 1 or reward == 0:
            if reward == 1:
                r = self.r1
            else:
                r = self.r0            
            self.A0 += np.dot(self.BaTAaI[self.a_max], self.Ba[self.a_max])
            self.b0 += np.dot(self.BaTAaI[self.a_max], self.ba[self.a_max])
            self.Aa[self.a_max] += np.dot(self.xa, self.xaT)
            self.AaI[self.a_max] = np.linalg.inv(self.Aa[self.a_max])
            self.Ba[self.a_max] += np.dot(self.xa, self.zT)
            self.BaT[self.a_max] = np.transpose(self.Ba[self.a_max])
            self.ba[self.a_max] += r * self.xa
            self.AaIba[self.a_max] = np.dot(self.AaI[self.a_max], self.ba[self.a_max])
            self.AaIBa[self.a_max] = np.dot(self.AaI[self.a_max], self.Ba[self.a_max])
            self.BaTAaI[self.a_max] = np.dot(self.BaT[self.a_max], self.AaI[self.a_max])            
            self.A0 += np.dot(self.z, self.zT) - np.dot(self.BaTAaI[self.a_max], self.Ba[self.a_max])
            self.b0 += r * self.z - np.dot(self.BaT[self.a_max], np.dot(self.AaI[self.a_max], self.ba[self.a_max]))
            self.A0I = np.linalg.inv(self.A0)
            self.beta = np.dot(self.A0I, self.b0)
            self.theta = self.AaIba - np.dot(self.AaIBa, self.beta)
                
        else:        
            pass       
 
              
    def reccomend(self, timestamp, user_features, articles):

        # Get user features
        article_len = len(articles)
        self.xaT = np.array([user_features])
        self.xa = np.transpose(self.xaT)
        
        # Get article features        
        index = [self.index[article] for article in articles]
        article_features_tmp = self.article_features[index]
        zaT = np.einsum('i,j', article_features_tmp.reshape(-1), user_features).reshape(article_len, 1, self.k)
        za = np.transpose(zaT, (0,2,1))
        zaT_tmp = zaT.reshape((article_len,self.k))
        za_tmp = np.transpose(zaT_tmp)
           
        # Initialize scores
        pt = np.zeros(article_len)
        
        # Perform matrix calculations
        for i, idx in enumerate(index):    
            A0IBaTAaIxa_tmp =  np.dot(self.A0I,np.dot(self.BaTAaI[idx], self.xa))
            A0Iza_tmp = np.dot(zaT_tmp[i], self.A0I)     
            sa_1_tmp = np.dot(A0Iza_tmp,za_tmp[:,i]) - 2*np.dot(zaT_tmp[i], A0IBaTAaIxa_tmp)
            sa_2_tmp = np.dot(np.dot(self.xaT, self.AaI[idx]),self.xa) + np.dot(np.dot(self.xaT,self.AaIBa[idx]),A0IBaTAaIxa_tmp)
            sa_tmp = sa_1_tmp + sa_2_tmp
            pt[i] = np.dot(zaT_tmp[i], self.beta) + np.dot(self.xaT,self.theta[idx]) + self.alpha * np.sqrt(sa_tmp)

        # Find maximum       
        max_index = np.argmax(pt)

        # Update variables
        self.z = za[max_index]
        self.zT = zaT[max_index]
        art_max = index[max_index]
        
        # Article index with largest UCB
        self.a_max = art_max
        
        return articles[max_index]

def set_articles(art):

    # Global variable
    global HybridUCBObj

    # Initialize class
    HybridUCBObj = HybridUCB()

    # Collect features from articles
    HybridUCBObj.set_articles(art)

    # Return output
    pass

def update(reward):

    # Update class
    HybridUCBObj.update(reward)

    # Return output
    pass 
 
def recommend(time, user_features, choices):

    #Return output
    return np.int64(HybridUCBObj.reccomend(time, user_features,choices))
   
