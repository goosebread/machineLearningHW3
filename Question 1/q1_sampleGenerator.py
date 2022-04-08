#Alex Yeh
#HW3
#Question 1 Sample Generator
#This script was run repeatedly until a sample set was generated that resulted in 
#the true pdf MAP classifer achieving between 10-20% error in the test set

import numpy as np

def makeDistributions():
    n=3 #number of dimensions
    
    #make 4 random cov matrices    
    A1=np.random.rand(n,n)-0.5
    S1=np.matmul(A1,A1.T) 
    A2=np.random.rand(n,n)-0.5
    S2=np.matmul(A2,A2.T)
    A3=np.random.rand(n,n)-0.5
    S3=np.matmul(A3,A3.T)
    A4=np.random.rand(n,n)-0.5
    S4=np.matmul(A4,A4.T)

    #plot means on cube edges
    dist = 0
    for S in [S1,S2,S3,S4]:
        dist += np.sum(np.sqrt(np.diag(S)))
    scale = dist/12
    m1 = scale * np.array([-1,-1,1])
    m2 = scale * np.array([-1,1,-1])
    m3 = scale * np.array([1,1,1])
    m4 = scale * np.array([1,-1,-1])

    #store mean/cov data to npz file
    with open('Q1_DistData.npz', 'wb') as f0:
        np.savez(f0,m1=m1,m2=m2,m3=m3,m4=m4,S1=S1,S2=S2,S3=S3,S4=S4)

def makeSamples(N,name):
    #N = number of Samples

    #uniform priors
    prior = 0.25
    distdata = np.load('Q1_DistData.npz')

    #generate true labels and samples
    A = np.random.rand(N,1)
    class1 = A<=prior 
    class2 = (A<=2*prior) & (A>prior) 
    class3 = (A<=3*prior) & (A>2*prior) 
    class4 = (A>3*prior) 

    trueClassLabels = class1+2*class2+3*class3+4*class4
    print("Class Priors")
    print("p(L=1) = "+str(np.sum(trueClassLabels==1)/N))
    print("p(L=2) = "+str(np.sum(trueClassLabels==2)/N))
    print("p(L=3) = "+str(np.sum(trueClassLabels==3)/N))
    print("p(L=4) = "+str(np.sum(trueClassLabels==4)/N))

    x1 = np.random.multivariate_normal(distdata['m1'],distdata['S1'], N)
    x2 = np.random.multivariate_normal(distdata['m2'],distdata['S2'], N)
    x3 = np.random.multivariate_normal(distdata['m3'],distdata['S3'], N)
    x4 = np.random.multivariate_normal(distdata['m4'],distdata['S4'], N)

    #class1,class2,class3,class4 are mutually exclusive and collectively exhaustive
    samples = class1*x1 + class2*x2 + class3*x3 + class4*x4

    #store true labels and samples
    with open(name+str(N)+'_Labels.npy', 'wb') as f1:
        np.save(f1, trueClassLabels)
    with open(name+str(N)+'_Samples.npy', 'wb') as f2:
        np.save(f2, samples)

#run script
if __name__ == '__main__':
    makeDistributions()
    makeSamples(100,"D_train")
    makeSamples(200,"D_train")
    makeSamples(500,"D_train")
    makeSamples(1000,"D_train")
    makeSamples(2000,"D_train")
    makeSamples(5000,"D_train")
    makeSamples(100000,"D_test")
