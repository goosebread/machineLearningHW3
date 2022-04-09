#Alex Yeh
#HW3
#Question 2 Sample Generator
#This script was run repeatedly until a sample set was generated that resulted in 
#the true pdf MAP classifer achieving between 10-20% error in the test set

import numpy as np

def makeDistributions():
    n=2 #number of dimensions
    
    #make 4 random cov matrices    
    A1=np.random.rand(n,n)-0.5
    S1=np.matmul(A1,A1.T) 
    A2=np.random.rand(n,n)-0.5
    S2=np.matmul(A2,A2.T)
    A3=np.random.rand(n,n)-0.5
    S3=np.matmul(A3,A3.T)
    A4=np.random.rand(n,n)-0.5
    S4=np.matmul(A4,A4.T)

    scale = 0.6
    m1 = scale * np.array([1,1])
    m2 = scale * np.array([1,-1])
    m3 = scale * np.array([-1,1])
    m4 = scale * np.array([-1,-1])

    #priors
    #3 dividers results in 4 intervals
    reg_factor = 0.3 #bias towards uniform prior
    p = np.random.rand(4)
    p[3] = 1
    p.sort()
    p[1:4] = p[1:4]-p[0:3]
    p = (1-reg_factor)*p+reg_factor*0.25
    print("Class Priors "+str(p))
    print(np.sum(p))

    #store mean/cov data to npz file
    with open('Q2_DistData.npz', 'wb') as f0:
        np.savez(f0,m1=m1,m2=m2,m3=m3,m4=m4,S1=S1,S2=S2,S3=S3,S4=S4,p=p)

def makeSamples(N,name):
    #N = number of Samples

    distdata = np.load('Q2_DistData.npz')
    priors = distdata['p']

    #generate and samples
    A = np.random.rand(N,1)
    class1 = A<=priors[0]
    class2 = (A<=priors[0]+priors[1]) & (A>priors[0]) 
    class3 = (A<=priors[0]+priors[1]+priors[2]) & (A>priors[0]+priors[1]) 
    class4 = (A>priors[0]+priors[1]+priors[2]) 

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

    #store samples only
    with open(name+str(N)+'_Samples.npy', 'wb') as f2:
        np.save(f2, samples)

#run script
if __name__ == '__main__':
    makeDistributions()

    makeSamples(10,"Samples")
    makeSamples(100,"Samples")
    makeSamples(1000,"Samples")
    makeSamples(10000,"Samples")


