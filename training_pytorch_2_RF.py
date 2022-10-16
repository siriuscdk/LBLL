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

"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.utils import plot_model
#from keras.utils.vis_utils import plot_model
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.models import load_model
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split
"""
import h5py

from numpy.random import seed

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.sampler import SubsetRandomSampler

from sklearn import metrics
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load


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
        #x = self.fc2(x)
        #x = F.relu(x)
        x = self.fc3(x)
        #output = F.log_softmax(x, dim=self.n_class)
        return x


def construct_dataset(benchpath, data_X, data_Y):
    num_train=0
    print (benchpath)
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
        #print (Xs.shape, Ys.shape)
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
    #seed(1)
    #set_random_seed(2)
    #tf.random.set_seed(3)
    #benchpath="./benchset"
    #all_data_X=[]
    #all_data_Y=[]
    #all_data_X = np.array([]).reshape(0, DF*12+3)
    #all_data_Y=np.array([]).reshape(0, 3)
    all_results={}
    num_classes = 3
    # classes=["LATCH_L0", "LATCH_L1", "LATCH_DD", "LATCH_LD"]
    classes = ["LATCH_NLD", "LATCH_LD"]

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

    #print (file)
    #x = re.findall(r"^\.\/PO_all_dataset_every_benchmark\\([A-Za-z0-9]+)+_dataset\.hdf5", file)
    #print (x)
    #benchname=x[0]
    input_dim=DF*5+5+2+1
    n_class=2
    valid_size=0.2
    batch_size=32

    data_X = np.array([]).reshape(0, input_dim)
    data_Y = np.array([]).reshape(0, n_class)


    data_X, data_Y=construct_dataset(benchpath, data_X, data_Y)
    print (data_X.shape)
    print (data_Y.shape)

    data_Y = np.argmax(data_Y, axis=1)

    X_train, X_val, y_train, y_val = train_test_split(data_X, data_Y, test_size =valid_size, random_state = 42)

    #Create a svm Classifier
    clf = svm.SVC(kernel='poly') # Linear Kernel

    # Train the model using the training sets
    clf.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(X_val)
    print("Accuracy:", metrics.accuracy_score(y_val, y_pred))
    dump(clf, './best_models/SVM_model_2')

    # Create Decision Tree classifer object
    clf = DecisionTreeClassifier()

    # Train Decision Tree Classifer
    clf = clf.fit(X_train,y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(X_val)

    print("Accuracy:", metrics.accuracy_score(y_val, y_pred))
    dump(clf, './best_models/DT_model_2')

    # Create Decision Tree classifer object
    clf = RandomForestClassifier()

    # Train Decision Tree Classifer
    clf = clf.fit(X_train,y_train)
    dump(clf, './best_models/RF_model_2_best_new')

    # Predict the response for test dataset
    y_pred = clf.predict(X_val)
    print("Accuracy:", metrics.accuracy_score(y_val, y_pred))

    #print ("search file")
    testbenchpath = "./all_testing_sets_2class"
    all_results = {}
    for idx, file in enumerate(glob.glob(testbenchpath + '/*')):
        print (file)
        x = re.findall(r"^\.\/all_testing_sets_2class\\([A-Za-z0-9]+)+_1_dataset\.hdf5", file)
        correct = 0
        with h5py.File(file, 'r') as hf:
            # X_train = hf['X_train'][:1]
            # Y_train = hf['Y_train'][:1]
            X_test = hf['X_train'][:]
            Y_test = hf['Y_train'][:]

        Y_test_label = np.argmax(Y_test, axis=1)

        # Predict the response for test dataset
        y_pred = clf.predict(X_test)

        test_acc=metrics.accuracy_score(Y_test_label, y_pred)
        print("Accuracy:", test_acc)
        all_results[x[0]] = all_results.get(x[0], []) + [test_acc, len(Y_test_label)*(1-test_acc), len(Y_test_label)]
        #print (X_test)


    print (all_results)
    #return data_X, data_Y
    f = open("RF_10datasets_cweights_simple_best_MLP_model_pytorch_2_testing_result_pytorch_new_new.txt", "w")
    for key, val in all_results.items():
        f.write(f'{key}:{val}\n')

    f.close()


#specific_bench='s9234'
#benchpath="./benchset/iscas89/train"s
#benchpath="./benchset/iscas89/test"
seed(1)

#tf.random.set_seed(3)

benchpath="./all_training_sets_2class"
all_bench='s'

DB = 1  # depth for backward (towards inputs)
DF = 1  # depth for forward (towards outputs)
#count_visited_FF = True

generate_dataset(all_bench, benchpath, DB, DF)









