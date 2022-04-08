import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal




#function (data, num gaussians, priors, mus, covs)
def evalPosterior(data, nGaussians, priors, mus, covs):
    NSamples = data.shape[0]
    pxgivenL = np.zeros(nGaussians,NSamples)
    pxandL = np.zeros(nGaussians)
    for n in range(nGaussians):
        mvn = multivariate_normal(mus[n],covs[n])
        pxgivenL[n,:] = mvn.pdf(data)
        pxandL[n,:] = pxgivenL[n,:]*priors[n]

        #normalize to make up for missing constant factor (px)
        pLgivenx = pxandL/np.sum(pxandL,axis=0)

        #normalized class frequency
        newPriors = np.average(pLgivenx,axis=1)

        #weighted average of samples
        weights = (pLgivenx.T/np.sum(pLgivenx,axis=1)).T
        newMu = np.matmul(weights,data)

        #calculate new covariance matrix
        #formula from EM_GMM matlab file
        for l in range(nGaussians):
            v = data-newMu[l]
            u = np.multiply(np.tile(weights[l],(dims,1)).T,v)
            newCovs[n] = np.matmul(v.T,u) + regWeight*np.eye(dims) # adding a small regularization term


mvn0b = multivariate_normal(m02,C02)
mvn1 = multivariate_normal(m1,C1)

#classify samples
samples = np.load(open('D_validate10000_Samples.npy', 'rb'))

pxgivenL0a = np.array([mvn0a.pdf(samples)]).T
pxgivenL0b = np.array([mvn0b.pdf(samples)]).T
pxgivenL1 = np.array([mvn1.pdf(samples)]).T

#likelihood ratio to be compared against gamma
ratio = pxgivenL1/(0.5*pxgivenL0a+0.5*pxgivenL0b) 

#construct array of gaussians
#return log likelihood of generating data (weighted likelihood from each gaussian)


#minimize to do numerical optimization. 