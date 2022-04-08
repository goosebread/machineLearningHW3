#def evalPosterior(data, nGaussians, priors, mus, covs):
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

dims = 2
regWeight = 1e-5
nGaussians=4
delta = 1e-9#convergence

data = np.load(open('Samples100_Samples.npy', 'rb'))
NSamples = data.shape[0]
print (NSamples)

#initial guess...........................................................................
priors = np.ones(nGaussians)/nGaussians
#choose random points for mu
mus = np.zeros((nGaussians,dims))
for n in range(nGaussians):
    rind = np.random.randint(NSamples)
    mus[n,:] = data[rind]

#assign each sample to nearest mean
distances = np.zeros((NSamples,nGaussians))
for n in range(nGaussians):
    distances[:,n] = np.linalg.norm((data-mus[n]),ord=2,axis=1)

#nearest mean for each sample
labelarray = np.argmin(distances,axis=1)

#sample covariances with guessed assigned groups
covs = np.zeros((nGaussians,dims,dims))
for n in range(nGaussians):
    sample_group = data[labelarray==n]
    covs[n,:,:] = np.cov(sample_group.T)

#loop until convergence....................................................................
converged = False
newCovs = np.zeros((nGaussians,dims,dims))
while (~converged):
    print("iteration")
    #get posteriors for sample data
    pxgivenL = np.zeros((nGaussians,NSamples))
    pxandL = np.zeros((nGaussians,NSamples))
    for n in range(nGaussians):
        mvn = multivariate_normal(mus[n],covs[n])
        pxgivenL[n,:] = mvn.pdf(data)
        pxandL[n,:] = pxgivenL[n,:]*priors[n]
    #normalize to make up for missing constant factor (px)
    pLgivenx = pxandL/np.sum(pxandL,axis=0)

    #update parameters
    #normalized class frequency
    newPriors = np.average(pLgivenx,axis=1)

    #weighted average of samples
    weights = (pLgivenx.T/np.sum(pLgivenx,axis=1)).T
    newMu = np.matmul(weights,data)
    
    #formula from EM_GMM matlab file
    for n in range(nGaussians):
        v = data-newMu[n]
        u = np.multiply(np.tile(weights[n],(dims,1)).T,v)
        newCovs[n,:,:] = np.matmul(v.T,u) + regWeight*np.eye(dims) # adding a small regularization term

    #check convergence
    diff = np.linalg.norm((priors-newPriors),ord=2)+np.linalg.norm((mus-newMu))+np.linalg.norm((covs-newCovs))
    print(diff*1000)
    covs = newCovs
    mus = newMu
    priors = newPriors

    if diff<delta:
        converged=True
        break

#evaluate fitting


#consider plotting


print(mus)