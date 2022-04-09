#def evalPosterior(data, nGaussians, priors, mus, covs)
from gettext import ngettext
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.covariance import log_likelihood

#this function partitions a dataset into a train and test group 
#according to the principles of k-fold validation
def kPartition(all_samp, ki, k, test):
    N = all_samp.shape[0]
    idx_low = int(np.floor(ki/k*N))
    idx_high = int(np.floor((ki+1)/k*N))
    if test:
        return all_samp[idx_low:idx_high]
    else:
        return np.concatenate([all_samp[0:idx_low],all_samp[idx_high:N]])

#get pdf value for sample from mixture of gaussian pdfs
def evalGMM(data,priors,mus,covs):
    gmm = np.zeros(data.shape[0])
    nGaussians = len(priors)
    for n in range(nGaussians):
        mvn = multivariate_normal(mus[n],covs[n])
        gmm += mvn.pdf(data)*priors[n]
    return gmm

def contourGMM(alpha,mu,Sigma,rangex1,rangex2):
    x1Grid = np.linspace(np.floor(rangex1[0]),np.ceil(rangex1[1]),101)
    x2Grid = np.linspace(np.floor(rangex2[0]),np.ceil(rangex2[1]),91)
    [h,v] = np.meshgrid(x1Grid,x2Grid)
    h=np.reshape(h,(101*91,1))
    v=np.reshape(v,(101*91,1))
    GMM = (evalGMM(np.concatenate([h,v],axis=1),alpha, mu, Sigma))
    zGMM = np.reshape(GMM,(91,101))
    return x1Grid,x2Grid,zGMM

def plotGaussians(data, priors, mus, covs):
    fig,ax = plt.subplots()
    ax.scatter(data[:,0],data[:,1],marker='x',color='b')
    ax.scatter(mus[:,0],mus[:,1],marker='o',color='r')
    ax.set_title('Data and Estimated GMM Contours')
    ax.set_aspect('equal')

    rangex0 = [np.min(data[:,0]),np.max(data[:,0])]
    rangex1 = [np.min(data[:,1]),np.max(data[:,1])]

    [x1Grid,x2Grid,zGMM] = contourGMM(priors,mus,covs,rangex0,rangex1)
    ax.contour(x1Grid,x2Grid,zGMM); 
    plt.show()




k=10
dims = 2
regWeight = 1e-6
maxGaussians=6
delta = 1e-4#convergence
nesting_threshold = 0.01
complexity_difference_threshold = 0.01

train_sets = ["Samples100","Samples1000","Samples10000"]

for s in range(len(train_sets)):
    train_set = train_sets[s]
    print(train_set)
    samples = np.load(open(train_set+'_Samples.npy', 'rb'))

    averageNLLS = np.zeros(maxGaussians)
    for nGaussians in range(1,maxGaussians+1):
        print(nGaussians)
            
        nlls =np.zeros(k)
        nested = np.zeros(k)

        for ki in range(k):
            train_data = kPartition(samples, ki, k, False)
            NSamples = train_data.shape[0]
            #initial guess...........................................................................
            priors = np.ones(nGaussians)/nGaussians
            #choose random points for mu
            mus = np.zeros((nGaussians,dims))
            rind = np.random.randint(NSamples)
            #this is still random but guarantees no repeats, assuming samples are iid
            for n in range(nGaussians):
                mus[n,:] = train_data[(rind+n)%nGaussians]

            #assign each sample to nearest mean
            distances = np.zeros((NSamples,nGaussians))
            for n in range(nGaussians):
                distances[:,n] = np.linalg.norm((train_data-mus[n]),ord=2,axis=1)

            #nearest mean for each sample
            labelarray = np.argmin(distances,axis=1)

            #sample covariances with guessed assigned groups
            covs = np.zeros((nGaussians,dims,dims))
            for n in range(nGaussians):
                sample_group = train_data[labelarray==n]
                covs[n,:,:] = np.cov(sample_group.T) + regWeight*np.eye(dims)

            #loop until convergence....................................................................
            converged = False
            newCovs = np.zeros((nGaussians,dims,dims))
            #iteration algorithm from EMforGMM matlab file provided. 
            while (~converged):
                #print("iteration")
                #get posteriors for sample data
                pxgivenL = np.zeros((nGaussians,NSamples))
                pxandL = np.zeros((nGaussians,NSamples))
                for n in range(nGaussians):
                    mvn = multivariate_normal(mus[n],covs[n])
                    pxgivenL[n,:] = mvn.pdf(train_data)
                    pxandL[n,:] = pxgivenL[n,:]*priors[n]
                #normalize to make up for missing constant factor (px)
                pLgivenx = pxandL/np.sum(pxandL,axis=0)

                #update parameters
                #normalized class frequency
                newPriors = np.average(pLgivenx,axis=1)

                #weighted average of samples
                weights = (pLgivenx.T/np.sum(pLgivenx,axis=1)).T
                newMu = np.matmul(weights,train_data)
                
                #formula from EM_GMM matlab file
                for n in range(nGaussians):          
                    v = train_data-newMu[n]
                    u = np.multiply(np.tile(weights[n],(dims,1)).T,v)
                    newCovs[n,:,:] = np.matmul(v.T,u) + regWeight*np.eye(dims)

                #check convergence
                diff = np.linalg.norm((priors-newPriors),ord=2)+np.linalg.norm((mus-newMu))+np.linalg.norm((covs-newCovs))
                #print(diff*1000)
                covs = newCovs
                mus = newMu
                priors = newPriors

                if diff<delta:
                    converged=True
                    break

            #evaluate fitting ...............................................................................
            test_data = kPartition(samples, ki, k, True)
            nTestSamples = test_data.shape[0]
            #use final mus, covs, priors to make gaussians and evaluate probability of seeing validation partition
            pxgivenw = evalGMM(test_data,priors,mus,covs)
            neglogLikelihood = -np.sum(np.log(pxgivenw))
            nlls[ki] = neglogLikelihood

            #true if any of the priors are below a threshold
            nested[ki] = not(np.product(priors>nesting_threshold))
            
            #just plot one
            if ki==9:
                plotGaussians(test_data,priors,mus,covs)

        averageNLL = 0
        if np.sum(nested)==0:
            averageNLL = np.average(nlls)
        else: 
            print("Nested Model Warning "+str(nGaussians)+ " Gaussian Model")
            #only count the models that aren't considered nested
            #averageNLL = np.average(nlls)
            averageNLL = np.sum(np.multiply(nlls,1-nested))/np.sum(1-nested)
        averageNLLS[nGaussians-1] = averageNLL#index starts at 0

    min_nll = averageNLLS[0]
    selectedModel = 1
    for m in range(1,len(averageNLLS)):
        if((min_nll-averageNLLS[m])/averageNLLS[m]<complexity_difference_threshold):
            break
        else:
            min_nll = averageNLLS[m]
            selectedModel = m+1
    print(averageNLLS)
    print(selectedModel)