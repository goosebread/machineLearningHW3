# Alex Yeh
# Question 2 Part A - 2,3

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.metrics import confusion_matrix
from matplotlib.colors import LinearSegmentedColormap

def runPartA(lossMatrix,title1,title2):

    #true data distribution is known for this ERM classifier
    distdata = np.load('Q1_DistData.npz')

    mvn1 = multivariate_normal(distdata['m1'],distdata['S1'])
    mvn2 = multivariate_normal(distdata['m2'],distdata['S2'])
    mvn3 = multivariate_normal(distdata['m3'],distdata['S3'])
    mvn4 = multivariate_normal(distdata['m4'],distdata['S4'])

    samples = np.load(open('D_test100000_Samples.npy', 'rb'))

    pxgivenL1 = mvn1.pdf(samples)
    pxgivenL2 = mvn2.pdf(samples)
    pxgivenL3 = mvn3.pdf(samples)
    pxgivenL4 = mvn4.pdf(samples)

    #class priors are given and are uniform
    prior = 0.25

    #p(x) can usually be factored out as a constant 
    #but we need to estimate minimum expected risk in this problem
    #so it's necessary to have the real class posteriors instead of something proportional
    px = (pxgivenL1+pxgivenL2+pxgivenL3+pxgivenL4)*prior
    pL1givenx = pxgivenL1 * prior / px
    pL2givenx = pxgivenL2 * prior / px
    pL3givenx = pxgivenL3 * prior / px
    pL4givenx = pxgivenL4 * prior / px

    #4xN matrix, each col represents probabilities for one sample
    P = np.stack((pL1givenx,pL2givenx,pL3givenx,pL4givenx))

    #Loss matrix for minimum total error
    L = lossMatrix

    #Risk vectors for each sample, 4xN matrix
    R = np.matmul(L,P)
    
    #Make Decisions based on minimum risk
    Decisions = np.array([np.argmin(R, axis=0)])+1

    #Estimate minimum expected risk for using 100k samples
    minRisks = R.min(axis=0)
    expectedMinRisk = np.average(minRisks)
    print("Minimum Expected Risk = "+str(expectedMinRisk))

    #calculate confusion matrix (using sklearn library)
    trueLabels = np.load(open('D_test100000_Labels.npy', 'rb')).T
    CM = confusion_matrix((trueLabels-1)[0], (Decisions-1)[0], normalize = 'true')
    print("Confusion Matrix: ")
    print(CM)

    doVisualization(samples, trueLabels, Decisions, title1, title2)

#Part 3 Visualizations
#separate by true label
def doVisualization(samples, trueLabels, Decisions, title1, title2):
    correctDecision = trueLabels==Decisions
    print("Measured Error = "+str(1-np.average(correctDecision)))
    data = np.concatenate((samples,trueLabels.T,Decisions.T,correctDecision.T),axis=1)

    #this filtering scheme requires a reshape to return to 2d matrix representation
    data1 = data[np.argwhere(data[:,3]==1),:]
    data1 = np.reshape(data1,(data1.shape[0],data1.shape[2]))
    data2 = data[np.argwhere(data[:,3]==2),:]
    data2 = np.reshape(data2,(data2.shape[0],data2.shape[2]))
    data3 = data[np.argwhere(data[:,3]==3),:]
    data3 = np.reshape(data3,(data3.shape[0],data3.shape[2]))
    data4 = data[np.argwhere(data[:,3]==4),:]
    data4 = np.reshape(data4,(data4.shape[0],data4.shape[2]))

    #plot Decisions vs Actual Label

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    cmap1 = LinearSegmentedColormap.from_list('4class', [(1, 0, 0), (0, 1, 0), (1, 0, 1), (1, 1, 0),(0, 0, 1)])
    legend_elements1 = [ax.scatter([0], [0], marker = 'o',c = 1, label='Label = 1',cmap=cmap1,vmin=1,vmax=4),
                    ax.scatter([0], [0], marker = 's',c = 2,label='Label = 2',cmap=cmap1,vmin=1,vmax=4),
                    ax.scatter([0], [0], marker = 'v',c = 3,label='Label = 3',cmap=cmap1,vmin=1,vmax=4),
                    ax.scatter([0], [0], marker = 'x',c = 4,label='Label = 4',cmap=cmap1,vmin=1,vmax=4)]
    l1=ax.scatter(data1[:,0],data1[:,1],zs=data1[:,2],marker = 'o',c = data1[:,4],label='Label = 1',cmap=cmap1,vmin=1,vmax=4)
    l2=ax.scatter(data2[:,0],data2[:,1],zs=data2[:,2],marker = 's',c = data2[:,4],label='Label = 2',cmap=cmap1,vmin=1,vmax=4)
    l3=ax.scatter(data3[:,0],data3[:,1],zs=data3[:,2],marker = 'v',c = data3[:,4],label='Label = 3',cmap=cmap1,vmin=1,vmax=4)
    l4=ax.scatter(data4[:,0],data4[:,1],zs=data4[:,2],marker = 'x',c = data4[:,4],label='Label = 4',cmap=cmap1,vmin=1,vmax=4)

    ax.set_title(title1)
    ax.legend(handles=legend_elements1,title="Shape = True Label\nColor = Classifier Decision")


    #plot Error vs Actual Label

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(projection='3d')

    cmap2 = LinearSegmentedColormap.from_list('redTransparentGreen', [(1, 0, 0, 1), (0.5, 1, 0.5, 0.1)])
    legend_elements2 = [ax.scatter([0], [0], marker = 'o',c = 1, label='Label = 1',cmap=cmap2,vmin=0,vmax=1),
                    ax.scatter([0], [0], marker = 's',c = 1,label='Label = 2',cmap=cmap2,vmin=0,vmax=1),
                    ax.scatter([0], [0], marker = 'v',c = 1,label='Label = 3',cmap=cmap2,vmin=0,vmax=1),
                    ax.scatter([0], [0], marker = 'x',c = 1,label='Label = 4',cmap=cmap2,vmin=0,vmax=1)]

    l12=ax2.scatter(data1[:,0],data1[:,1],zs=data1[:,2],marker = 'o',c = data1[:,5], label='Label = 1',cmap=cmap2,vmin=0,vmax=1)
    l22=ax2.scatter(data2[:,0],data2[:,1],zs=data2[:,2],marker = 's',c = data2[:,5], label='Label = 2',cmap=cmap2,vmin=0,vmax=1)
    l32=ax2.scatter(data3[:,0],data3[:,1],zs=data3[:,2],marker = 'v',c = data3[:,5], label='Label = 3',cmap=cmap2,vmin=0,vmax=1)
    l42=ax2.scatter(data4[:,0],data4[:,1],zs=data4[:,2],marker = 'x',c = data4[:,5], label='Label = 4',cmap=cmap2,vmin=0,vmax=1)
    
    ax2.set_title(title2)
    lg = ax2.legend(handles=legend_elements2,title="Red Marker = Incorrect Classification")
    for i in range(4):
        lg.legendHandles[i].set_alpha(1)
    plt.show()

if __name__ == '__main__':
    lossMatrixA = np.ones((4,4)) - np.eye(4)
    runPartA(lossMatrixA,'True PDF Classifier Decision vs True Label','True PDF Classifier Error vs True Label')
