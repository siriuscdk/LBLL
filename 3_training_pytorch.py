import sys
import glob
import math
import re, random
from Ntk_Struct_PO_cmu import *
from Ntk_Parser_PO_cmu import *
from fflatch_only_graph_PO import *
import numpy as np
import networkx as nx
import collections
import h5py
from sklearn.model_selection import train_test_split
from numpy.random import seed
import h5py
from numpy.random import seed
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.sampler import SubsetRandomSampler


class MLP(nn.Module):
    def __init__(self, input_dim, n_class):
        super(MLP, self).__init__()
        self.n_class=n_class

        self.fc1=nn.Linear(input_dim, 100)
        #self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, n_class)

    def forward(self, x):
        x=self.fc1(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


def construct_dataset(benchpath, data_X, data_Y):
    num_train=0
    for idx, file in enumerate(glob.glob(benchpath + '/*')):
        num_train+=1

        with h5py.File(file, 'r') as hf:
            X = hf['X_train'][:]
            Y = hf['Y_train'][:]

        data_X = np.vstack((data_X, X))
        data_Y = np.vstack((data_Y, Y))
    print (num_train)

    return data_X, data_Y

def get_accuracy(preds, Ys):
    max_preds = preds.argmax(dim=1, keepdim=True)
    numcorrect=max_preds.squeeze(1).eq(Ys)
    return numcorrect.sum()/torch.FloatTensor([Ys.shape[0]])

def train(model, train_loader, optimizer, criterion):
    model.train()
    epoch_loss = 0
    epoch_acc = 0

    for Xs, Ys in train_loader:
        optimizer.zero_grad()
        preds=model(Xs.float())
        preds = preds.view(-1, preds.shape[-1])
        Ys=Ys.view(-1)
        loss = criterion(preds, Ys)
        acc=get_accuracy(preds, Ys)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(train_loader), epoch_acc / len(train_loader)

def evaluate(model, validation_loader, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()
    with torch.no_grad():
        for Xs, Ys in validation_loader:
            preds = model(Xs.float())
            preds = preds.view(-1, preds.shape[-1])
            Ys = Ys.view(-1)
            loss = criterion(preds, Ys)
            acc = get_accuracy(preds, Ys)
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(validation_loader), epoch_acc / len(validation_loader)

def generate_dataset(specific_bench,benchpath,DB, DF):
    seed(12)
    torch.manual_seed(12)
    torch.cuda.manual_seed(12)
    torch.cuda.manual_seed_all(12)
    np.random.seed(12)
    random.seed(12)

    # torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
    is_cuda = torch.cuda.is_available()

    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print (is_cuda)
    input_dim=DF*5+5+2+1
    n_class=3
    valid_size=0.2
    batch_size=32

    data_X = np.array([]).reshape(0, input_dim)
    data_Y = np.array([]).reshape(0, n_class)


    data_X, data_Y=construct_dataset(benchpath, data_X, data_Y)
    print (data_X.shape)
    print (data_Y.shape)

    data_Y = np.argmax(data_Y, axis=1)

    X_train, X_val, y_train, y_val = train_test_split(data_X, data_Y, test_size =valid_size, random_state = 42)

    train_data=torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_data = torch.utils.data.TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,shuffle=True)
    validation_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size,shuffle=True)

    model=MLP(input_dim, n_class)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    weights=[]
    fic = open(f"class_weights_{n_class}.txt", "r")
    for line in fic:
        weights.append(float(line.rstrip()))
    weights=torch.FloatTensor(weights)
    print (weights)
    fic.close()
    criterion = nn.CrossEntropyLoss(weight=weights)

    num_epoch = 100
    best_valid_acc = float('-inf')
    for epoch in range(num_epoch):
        train_loss, train_acc= train(model, train_loader, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, validation_loader, criterion)

        if valid_acc>best_valid_acc:
            best_valid_acc=valid_acc
            torch.save(model.state_dict(), f"./best_models/8datasets_cweights_simple_best_MLP_model_pytorch_{n_class}.pt")

        print(f'Epoch: {epoch + 1}')
        print(f'Train Loss:{train_loss :.4f} | Train Acc: {train_acc :.4f}%')
        print(f'Val Loss:{valid_loss :.4f} | Val Acc: {valid_acc :.4f}%')





seed(1)
benchpath="./all_training_sets_3class"
all_bench='s'

DB = 1  # depth for backward (towards inputs)
DF = 1  # depth for forward (towards outputs)

generate_dataset(all_bench, benchpath, DB, DF)









