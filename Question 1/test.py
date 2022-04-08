import numpy as np

name = "D_train100"
#samples = np.load(open(name+'_Samples.npy', 'rb'))
#labels = np.load(open(name+'_Labels.npy', 'rb'))
k=10
ki=9
validate = False
all_samp = np.load(open(name+'_Samples.npy', 'rb'))
all_lab = np.load(open(name+'_Labels.npy', 'rb'))
N = int(all_lab.size)
idx_low = int(np.floor(ki/k*N))
idx_high = int(np.floor((ki+1)/k*N))
if validate:
    ssamples = all_samp[idx_low:idx_high]
    slabels = all_lab[idx_low:idx_high]
else:
    slabels = np.concatenate([all_lab[0:idx_low],all_lab[idx_high:N]])
print(slabels.size)
print(all_lab)