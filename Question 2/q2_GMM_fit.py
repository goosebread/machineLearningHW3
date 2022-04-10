#Alex Yeh
#HW3 Question 2

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

#this function partitions a dataset into a train and test group 
#according to the principles of k-fold cross validation
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

#creates contour lines for GMM for plotting, from EMforGMM.m script
def contourGMM(alpha,mu,Sigma,rangex1,rangex2):
    x1Grid = np.linspace(np.floor(rangex1[0]),np.ceil(rangex1[1]),101)
    x2Grid = np.linspace(np.floor(rangex2[0]),np.ceil(rangex2[1]),91)
    [h,v] = np.meshgrid(x1Grid,x2Grid)
    h=np.reshape(h,(101*91,1))
    v=np.reshape(v,(101*91,1))
    GMM = (evalGMM(np.concatenate([h,v],axis=1),alpha, mu, Sigma))
    zGMM = np.reshape(GMM,(91,101))
    return x1Grid,x2Grid,zGMM

#plots sample data and GMM contours, from EMforGMM.m script
#GMM means are also plotted
def plotGaussians(data, priors, mus, covs):
    fig,ax = plt.subplots()

    rangex0 = [np.min([np.min(data[:,0]),np.min(mus[:,0])]),np.max([np.max(data[:,0]),np.max(mus[:,0])])]
    rangex1 = [np.min([np.min(data[:,1]),np.min(mus[:,1])]),np.max([np.max(data[:,1]),np.max(mus[:,1])])]
    [x1Grid,x2Grid,zGMM] = contourGMM(priors,mus,covs,rangex0,rangex1)
    ax.contour(x1Grid,x2Grid,zGMM,zorder=1)

    l1=ax.scatter(data[:,0],data[:,1],marker='x',color='b', label = "Test Data",zorder=0)
    l2=ax.scatter(mus[:,0],mus[:,1],marker='o',color='r', label = "Estimated GMM Means",zorder=2)
    ax.set_title('Data and Estimated GMM Contours')
    ax.set_aspect('equal')
    ax.legend(handles = [l1,l2])
    plt.show()

#algorithm from EMforGMM.m file
#returns an initial guess for model parameters
def initializeModel(train_data,nGaussians,regWeight):
    dims = 2
    NSamples = train_data.shape[0]

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
        if len(sample_group)<2:
            #just consider all of the points in the group if the mean is really bad
            sample_group = train_data

        covs[n,:,:] = np.cov(sample_group.T) + regWeight*np.eye(dims)
    return priors, mus, covs

#algorithm from EMforGMM.m file
#updates the model parameters for priors, means, and covariances
def loopIteration(train_data,priors,mus,covs,nGaussians,regWeight):
    dims = 2
    NSamples = train_data.shape[0]

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
    newMus = np.matmul(weights,train_data)

    newCovs = np.zeros((nGaussians,dims,dims))
    #formula from EM_GMM matlab file
    for n in range(nGaussians):          
        v = train_data-newMus[n]
        u = np.multiply(np.tile(weights[n],(dims,1)).T,v)
        newCovs[n,:,:] = np.matmul(v.T,u) + regWeight*np.eye(dims)

    return newPriors, newMus, newCovs

#get vector of NLL values for each model order for one training set
def runTrainingSet(train_set,silence,shuffle):
    k=10
    regWeight = 1e-4
    maxGaussians=6
    delta = 1e-4#convergence
    nesting_threshold = 0.01
    complexity_difference_threshold = 0.01
    if not(silence):
        print(train_set)
    samples = np.load(open(train_set+'_Samples.npy', 'rb'))
    if shuffle:
        np.random.shuffle(samples)
    averageNLLS = np.zeros(maxGaussians)
    for nGaussians in range(1,maxGaussians+1):
        if not(silence):
            print(nGaussians)
            
        nlls =np.zeros(k)
        nested = np.zeros(k)

        for ki in range(k):
            train_data = kPartition(samples, ki, k, False)
            #initial guess
            [priors, mus, covs] = initializeModel(train_data, nGaussians, regWeight)

            #loop until convergence
            converged = False
            while (~converged):
                #update values
                [newPriors, newMus, newCovs] = loopIteration(train_data,priors,mus,covs,nGaussians,regWeight)
                #check convergence
                diff = np.linalg.norm((priors-newPriors),ord=2)+np.linalg.norm((mus-newMus))+np.linalg.norm((covs-newCovs))
                covs = newCovs
                mus = newMus
                priors = newPriors
                if diff<delta:
                    converged=True
                    break

            #evaluate model
            test_data = kPartition(samples, ki, k, True)
            #use final mus, covs, priors to make gaussians and evaluate probability of seeing validation data
            pxgivenw = evalGMM(test_data,priors,mus,covs)
            neglogLikelihood = -np.sum(np.log(pxgivenw))
            nlls[ki] = neglogLikelihood

            #true if any of the priors are below a threshold
            nested[ki] = not(np.product(priors>nesting_threshold))
            
            #plot only last partition
            if ~silence & ki==9:
                plotGaussians(test_data,priors,mus,covs)

        averageNLL = 0
        if np.sum(nested)==0:
            averageNLL = np.average(nlls)
        else: 
            if not(silence):
                print("Nested Model Warning "+str(nGaussians)+ " Gaussian Model")
            #only count the models that aren't considered nested
            #try not to divide by zero
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
    return(selectedModel,averageNLLS)

#run script
if __name__ == '__main__':
    train_sets = ["Samples10","Samples100","Samples1000","Samples10000"]

    for s in range(len(train_sets)):
        [selectedModel,averageNLLS] = runTrainingSet(train_sets[s],False,False)
        print(averageNLLS)
        print(selectedModel)