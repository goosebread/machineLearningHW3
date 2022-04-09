#Original Code from pytorch tutorial
#modified by Alex Yeh

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import StepLR

class CustomDataset(Dataset):
    #assuming k>1, ki starts counting from partition 0
    #assuming k<N
    def __init__(self, name, ki, k, validate):
        all_samp = np.load(open(name+'_Samples.npy', 'rb'))
        all_lab = np.load(open(name+'_Labels.npy', 'rb'))
        N = all_lab.size
        idx_low = int(np.floor(ki/k*N))
        idx_high = int(np.floor((ki+1)/k*N))
        if validate:
            self.samples = all_samp[idx_low:idx_high]
            self.labels = all_lab[idx_low:idx_high]
        else:
            self.samples = np.concatenate([all_samp[0:idx_low],all_samp[idx_high:N]])
            self.labels = np.concatenate([all_lab[0:idx_low],all_lab[idx_high:N]])

    def __len__(self):
        return self.labels.size

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = int(self.labels[idx][0])
        label_vec = np.zeros(4)#4 classes hard coded
        label_vec[label-1]=1
        return sample, label_vec

#NN model for 2 Layer MLP with variable number of perceptrons in first hidden layer
class Net(nn.Module):
    def __init__(self,nPerceptrons):
        super(Net, self).__init__()
        #input data has 3 dimensions
        self.fc1 = nn.Linear(3, nPerceptrons)
        self.fc2 = nn.Linear(nPerceptrons,4)#4 possible output classes

    def forward(self, x):
        #first hidden layer of perceptrons
        x = self.fc1(x)
        #smooth ramp style activation function
        x = F.elu(x)
        #second/output layer
        x = self.fc2(x)
        output = F.softmax(x, dim=1)
        return output

#default method batch gradient descent
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss=0
    for batch_idx, (data, target) in enumerate(train_loader):

        #type casting to make things work
        data = data.float()
        target = target.float()

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.binary_cross_entropy(output, target) #minimize cross entropy loss
        train_loss+=loss.item()
        loss.backward()
        optimizer.step()
        
        #optional debugging information
        log=False
        if log:
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
                if args.dry_run:
                    break
            
    train_loss /= len(train_loader.dataset)
    return train_loss


def test(model, device, test_loader):
    model.eval()

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            #type casting to make things work
            data = data.float()
            target = target.float()

            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.binary_cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max probability
            label = target.argmax(dim=1, keepdim=True)  # get the index of the max probability
            correct += (pred==label).sum() #pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    #print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #    test_loss, correct, len(test_loader.dataset),
    #    100. * correct / len(test_loader.dataset)))

    #return error 
    return test_loss

def evaluate_n_Perceptrons(n,k,train_set):
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = False #not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': 20}#args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    #k-fold cross validation
    loss_sum = 0
    for ki in range(k):
        dataset_train = CustomDataset(train_set,ki,k,False)
        dataset_validate = CustomDataset(train_set,ki,k,True)

        train_loader = torch.utils.data.DataLoader(dataset_train,**train_kwargs)
        validate_loader = torch.utils.data.DataLoader(dataset_validate, **test_kwargs)

        #instantiate model
        model = Net(n).to(device)
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

        #breakpoint()

        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
        loss_old = -1     

        for epoch in range(1, 100):#args.epochs + 1):
            #honestly not sure if the batch size is shuffled or not. 
            loss = train(args, model, device, train_loader, optimizer, epoch)
            #loss = test(model, device, validate_loader)
            if(np.abs(loss-loss_old)<1e-12):
                break
            loss_old = loss
            scheduler.step()
            if epoch==99:
                print("Warning")

        #loss for fully trained model
        loss = test(model, device, validate_loader)
        loss_sum+=loss
        del model
    avg_loss = loss_sum/k
    return avg_loss

if __name__ == '__main__':
    #model order selection
    k=10
    exponents = range(2,14,1) #2 to 13
    n_values = np.power(2,exponents)
    results = pd.DataFrame(index = n_values)

    train_sets = ["D_train100","D_train200","D_train500","D_train1000","D_train2000","D_train5000"]
    for t in range(len(train_sets)):
        train_set = train_sets[t]
        #since we are using cross validation 
        losses = np.zeros(n_values.size)
        for i in range(n_values.size):
            losses[i] = evaluate_n_Perceptrons(n_values[i],k,train_set)
        m_idx = np.argmin(losses)
        results[train_set] = losses
        print("Training Set = "+train_set)
        print("Min Loss = "+str(losses[m_idx]))
        print("Number of Perceptrons = "+str(n_values[m_idx]))

    results.to_csv("Q1_Perceptron_Selection.csv")