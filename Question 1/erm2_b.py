# Alex Yeh
# Question 2 Part B

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.metrics import confusion_matrix
from erm2_a import runPartA

lossMatrixA = np.ones((3,3)) - np.eye(3)
lossMatrixB10 = np.matmul(np.ones((3,3)) - np.eye(3),np.diag((1,1,10)))
lossMatrixB100 = np.matmul(np.ones((3,3)) - np.eye(3),np.diag((1,1,100)))

runPartA(lossMatrixA,'ERM Decision vs True Label','ERM Error vs True Label')
print("Using Loss Matrix L10: ")
runPartA(lossMatrixB10,'ERM Decision vs True Label for L10','ERM Error vs True Label for L10')
print("Using Loss Matrix L100: ")
runPartA(lossMatrixB100,'ERM Decision vs True Label for L100','ERM Error vs True Label for L100')
