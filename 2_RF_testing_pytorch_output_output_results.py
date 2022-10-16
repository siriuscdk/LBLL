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
from sklearn.preprocessing import MinMaxScaler
from numpy.random import seed
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.sampler import SubsetRandomSampler
from joblib import dump, load
from sklearn import metrics
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


class MLP(nn.Module):
    def __init__(self, input_dim, n_class):
        super(MLP, self).__init__()
        self.n_class=n_class

        self.fc1=nn.Linear(input_dim, 100)
        self.fc3 = nn.Linear(100, n_class)

    def forward(self, x):
        x=self.fc1(x)
        x = F.relu(x)
        x = self.fc3(x)
        #output = F.log_softmax(x, dim=self.n_class)
        return x

def customgraph2networkx(netlist_graph):
    G = nx.DiGraph()
    for node in netlist_graph.object_list:
        if node.name not in G.nodes:
            G.add_node(node.name, type=netlist_graph.gateType_reverse[node.gate_type])
        else: # already added by edge
            G.nodes[node.name]['type'] = netlist_graph.gateType_reverse[node.gate_type]

        for innode in node.fan_in_node:
            G.add_edge(innode.name,node.name)

        for outnode in node.fan_out_node:
            G.add_edge(node.name,outnode.name)

    return G

def feature_normalization(cur_data_X):
    #print (cur_data_X)

    # fit scaler on training data
    norm = MinMaxScaler().fit(cur_data_X)

    # transform training data
    cur_data_X = norm.transform(cur_data_X)

    return cur_data_X

def add_q2latchname(benchname, filepath, q2latchname):
    fi=open(filepath+'/'+benchname+'_latchname2Q','r')
    for line in fi:
        #print (line.rstrip().split(':'))
        q, latchname=line.rstrip().split(':')
        #print (q, latchname)
        q2latchname[q]=latchname
    fi.close()

def add_delay(node, benchname, report_dir, q2latchname, seq_sig):
    latchname=q2latchname[node]
    PIflag=False # this latch connects to input

    f1=open(report_dir+latchname+'.from','r')
    f2 = open(report_dir + latchname + '.to', 'r')
    for line in f1:
        #if 'Endpoint' in line:
            #if seq_sig not in line: # remove the delay from input
                #print (node, line)

        if 'Required Time' in line:
            #print (line.rstrip().split(' '))
            fromreqtime=int(line.rstrip().split(' ')[-1])

        if 'Launch Clock' in line:
            #print (line.rstrip().split(' '))
            fromlclk=int(line.rstrip().split(' ')[-1])

        if 'Data Path' in line:
            #print (line.rstrip().split(' '))
            fromdelay=int(line.rstrip().split(' ')[-1])


    for line in f2:
        if 'Startpoint' in line:
            if seq_sig not in line:
                #print (node, line)
                PIflag=True

        if 'Required Time' in line:
            #print (line.rstrip().split(' '))
            toreqtime=int(line.rstrip().split(' ')[-1])

        if 'Launch Clock' in line:
            #print (line.rstrip().split(' '))
            tolclk=int(line.rstrip().split(' ')[-1])

        if 'Data Path' in line:
            todelay=int(line.rstrip().split(' ')[-1])

    f1.close()
    f2.close()
    if PIflag:
        todelay=0

    return fromdelay, fromreqtime-fromlclk, todelay, toreqtime-tolclk


def construct_dataset(curfile, benchpath, data_X, data_Y):
    num_train=0
    for idx, file in enumerate(glob.glob(benchpath + '/*')):
        if file!=curfile:
            num_train+=1

            with h5py.File(file, 'r') as hf:
                X = hf['X_train'][:]
                Y = hf['Y_train'][:]

            data_X = np.vstack((data_X, X))
            data_Y = np.vstack((data_Y, Y))
    #print (num_train)

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

def evaluate(model, validation_loader, criterion, all_results, benchname, num_classes):
    global misclassification
    epoch_loss = 0
    epoch_acc = 0

    allsoftmax_probs=[]
    probs = nn.Softmax(dim=1)

    confusion_matrix = [[0] * num_classes for i in range(num_classes)]

    model.eval()
    with torch.no_grad():
        for Xs, Ys in validation_loader:
            preds = model(Xs.float())
            preds = preds.view(-1, preds.shape[-1])
            predsoftmax = probs(preds)
            allsoftmax_probs.append(np.array(predsoftmax[0]))

            Ys = Ys.view(-1)

            preds_arg = np.argmax(preds, axis=1)
            confusion_matrix[Ys[0]][preds_arg[0]] += 1

            loss = criterion(preds, Ys)
            acc = get_accuracy(preds, Ys)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    print ("correct count:", epoch_acc, len(validation_loader))
    np_cm = np.array(confusion_matrix)
    print ('confusion matrix.')
    print (np_cm)
    all_results[benchname]=all_results.get(benchname, [])+[epoch_acc, len(validation_loader)]
    misclassification+=len(validation_loader)-epoch_acc
    return epoch_loss / len(validation_loader), epoch_acc / len(validation_loader), np.array(allsoftmax_probs)


def add_dataset(benchname, seq_sig, q2latchname,report_dir, DB, DF, G_fflatch_only, nxG, data_X, comb_data_X, data_Y, test_nodes):
    for node in G_fflatch_only:
        if not (G_fflatch_only.nodes[node]['type'] == 'LATCH_L0' or G_fflatch_only.nodes[node]['type'] == 'LATCH_L1' or G_fflatch_only.nodes[node]['type'] == 'LATCH_LD' or G_fflatch_only.nodes[node]['type'] == 'LATCH_DD'):
           continue

        #if len(G_fflatch_only.in_edges(node))==0:
            #continue

        tot_num_FI = len(G_fflatch_only.in_edges(node))
        tot_num_FO = len(G_fflatch_only.out_edges(node))

        visited = set() # add current node
        visited.add(node)

        # search backward
        bdepthidx = 0
        backl = []
        backq = collections.deque([node])

        while bdepthidx < DB:

            unseen_fanin_latch = 0
            seen_fanin_latch = 0
            unseen_fanin_ff=0
            seen_fanin_ff=0
            unseen_PIs = 0
            seen_PIs=0

            for _ in range(len(backq)):
                cur=backq.popleft()



                for inedge in G_fflatch_only.in_edges(cur):
                    innode=inedge[0]



                    if innode not in visited:
                        #if innode not in cur_visited:
                        if G_fflatch_only.nodes[innode]['type'] == 'IPT':
                            unseen_PIs += 1
                        elif G_fflatch_only.nodes[innode]['type'] == 'DFF':  # FF
                            unseen_fanin_ff+=1
                        elif G_fflatch_only.nodes[innode]['type'] == 'LATCH_L0':
                            #unseen_fanin_latch0+=1
                            unseen_fanin_latch+=1
                        elif G_fflatch_only.nodes[innode]['type'] == 'LATCH_L1':
                            #unseen_fanin_latch1+=1
                            unseen_fanin_latch += 1
                        elif G_fflatch_only.nodes[innode]['type'] == 'LATCH_LD':
                            #unseen_fanin_latchld+=1
                            unseen_fanin_latch += 1
                        elif G_fflatch_only.nodes[innode]['type'] == 'LATCH_DD':
                            #unseen_fanin_latchdd+=1
                            unseen_fanin_latch += 1
                        backq.append(innode)

                        #cur_visited.add(innode)
                        visited.add(innode)
                    else: # seen this node before
                        #print ("seen", bdepthidx)
                        if G_fflatch_only.nodes[innode]['type'] == 'IPT':
                            seen_PIs += 1
                        elif G_fflatch_only.nodes[innode]['type'] == 'DFF':  # FF
                            seen_fanin_ff+=1
                        elif G_fflatch_only.nodes[innode]['type'] == 'LATCH_L0':
                            #seen_fanin_latch0+=1
                            seen_fanin_latch+=1
                        elif G_fflatch_only.nodes[innode]['type'] == 'LATCH_L1':
                            #seen_fanin_latch1+=1
                            seen_fanin_latch += 1
                        elif G_fflatch_only.nodes[innode]['type'] == 'LATCH_LD':
                            #seen_fanin_latchld+=1
                            seen_fanin_latch += 1
                        elif G_fflatch_only.nodes[innode]['type'] == 'LATCH_DD':
                            #seen_fanin_latchdd+=1
                            seen_fanin_latch += 1
                        backq.append(innode)

            #visited |= cur_visited

            backl.append(unseen_PIs)
            backl.append(unseen_fanin_ff)
            backl.append(unseen_fanin_latch)
            bdepthidx+=1

        # search forward
        visited = set()  # add current node
        visited.add(node)
        fdepthidx = 0
        forwl = []
        forwq = collections.deque([node])
        while fdepthidx<DF:
            unseen_fanout_latch = 0
            seen_fanout_latch = 0
            unseen_fanout_ff=0
            seen_fanout_ff=0
            unseen_POs = 0
            seen_POs=0
            for _ in range(len(forwq)):
                cur=forwq.popleft()

                #tot_fanout+=len(G_fflatch_only.out_edges(cur))

                for outedge in G_fflatch_only.out_edges(cur):
                    outnode=outedge[1]
                    if outnode not in visited:
                        #if '_PO' in outnode:
                        if G_fflatch_only.nodes[outnode]['type'] == 'PO':
                            unseen_POs += 1
                        elif G_fflatch_only.nodes[outnode]['type'] == 'DFF':  # FF
                            unseen_fanout_ff+=1
                        elif G_fflatch_only.nodes[outnode]['type'] == 'LATCH_L0':
                            #unseen_fanout_latch0+=1
                            unseen_fanout_latch+=1
                        elif G_fflatch_only.nodes[outnode]['type'] == 'LATCH_L1':
                            #unseen_fanout_latch1+=1
                            unseen_fanout_latch += 1
                        elif G_fflatch_only.nodes[outnode]['type'] == 'LATCH_LD':
                            #unseen_fanout_latchld+=1
                            unseen_fanout_latch += 1
                        elif G_fflatch_only.nodes[outnode]['type'] == 'LATCH_DD':
                            #unseen_fanout_latchdd+=1
                            unseen_fanout_latch += 1

                        forwq.append(outnode)
                            #if not count_visited_FF:
                        visited.add(outnode)
                    else:
                        if G_fflatch_only.nodes[outnode]['type'] == 'PO':
                            seen_POs += 1
                        elif G_fflatch_only.nodes[outnode]['type'] == 'DFF':  # FF
                            seen_fanout_ff+=1
                        elif G_fflatch_only.nodes[outnode]['type'] == 'LATCH_L0':
                            #seen_fanout_latch0+=1
                            seen_fanout_latch+=1
                        elif G_fflatch_only.nodes[outnode]['type'] == 'LATCH_L1':
                            #seen_fanout_latch1+=1
                            seen_fanout_latch += 1
                        elif G_fflatch_only.nodes[outnode]['type'] == 'LATCH_LD':
                            #seen_fanout_latchld+=1
                            seen_fanout_latch += 1
                        elif G_fflatch_only.nodes[outnode]['type'] == 'LATCH_DD':
                            #seen_fanout_latchdd+=1
                            seen_fanout_latch += 1
                        forwq.append(outnode)

            forwl.append(unseen_fanout_ff)
            forwl.append(unseen_fanout_latch)

            fdepthidx+=1

        curFO=set()

        # detect self-loop of latch (M-S form a loop), 3 latch loop

        childFO=set()


        for outedge in G_fflatch_only.out_edges(node):
            outnode=outedge[1]
            curFO.add(outnode)

            for choutedge in G_fflatch_only.out_edges(outnode):
                choutnode=choutedge[1]
                childFO.add(choutnode)


        # compute the fraction
        fraction=0
        Sfraction=0


        curFI = set()
        parentFI = set()

        for inedge in G_fflatch_only.in_edges(node):
            innode = inedge[0]

            curFI.add(innode)

            parentFO=set([outedge[1] for outedge in G_fflatch_only.out_edges(innode)])

            if len(curFO & parentFO)>0:
                fraction+=1

            if len(childFO & parentFO)>0:
                Sfraction+=1

            for pinedge in G_fflatch_only.in_edges(innode):
                pinnode=pinedge[0]
                parentFI.add(pinnode)


        Sfraction2=0 # detect 2nd DD in trapezoid shape

        for outnode in curFO:
            childFI = set([inedge[0] for inedge in G_fflatch_only.in_edges(outnode)])

            if len(childFI & parentFI)>0:
                Sfraction2+=1




        latchloop=1 if len(childFO & curFI)>0 else 0




        backl.reverse()

        if tot_num_FI==0:

            tot_num_FI=1
        vector = backl + forwl + [fraction / tot_num_FI, Sfraction/tot_num_FI, Sfraction2/tot_num_FO,latchloop]


        # single path feature
        if len(G_fflatch_only.out_edges(node))==1 or len(G_fflatch_only.in_edges(node))==1:
            vector.append(1)
        else:
            vector.append(0)



        #print (node)
        if '_PO' in node:
            nodename=node[:-3]

        else:
            nodename=node
        fromdelay, fromdenominator, todelay, todenominator=add_delay(nodename, benchname, report_dir, q2latchname, seq_sig)
        vector.append(fromdelay)
        vector.append(todelay)


        # 13 self-loop feature
        all_fanout_self_loop=[]
        for outedge in G_fflatch_only.out_edges(node):
            outnode=outedge[1]
            child_selfloop=None
            child_num_fanin=len(G_fflatch_only.in_edges(outnode))
            for choutedge in G_fflatch_only.out_edges(outnode):
                choutnode = choutedge[1]
                if choutnode==outnode:
                    child_selfloop=True

            if child_selfloop:
                #print (G_fflatch_only.nodes[node]['type'], outnode, G_fflatch_only.out_edges(outnode))
                all_fanout_self_loop.append(round(1/child_num_fanin, 2))

        if all_fanout_self_loop:
            vector.append(max(all_fanout_self_loop))
        else: # none fanout has self-loop
            vector.append(0)

        data_X.append(vector)
        test_nodes.append(node)

        # add labels according to the latch's type
        if G_fflatch_only.nodes[node]['type'] == 'LATCH_L0':
            data_Y.append(np.array([1, 0]))
        elif G_fflatch_only.nodes[node]['type'] == 'LATCH_L1':
            data_Y.append(np.array([1, 0]))
        elif G_fflatch_only.nodes[node]['type'] == 'LATCH_DD':
            data_Y.append(np.array([1, 0]))
        elif G_fflatch_only.nodes[node]['type'] == 'LATCH_LD':
            data_Y.append(np.array([0, 1]))

def generate_dataset(specific_bench,benchpath,DB, DF, seq_sig):

    all_results={}

    global misclassification
    misclassification = 0


    for idx, filepath in enumerate(glob.glob(benchpath+'/*')):
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

        x = re.findall(r"^\.\/dataset_seed1_test_wo_remove_LD\\([A-Za-z0-9\_]+)$", filepath)
        benchname=x[0].split('_')[0]
        input_dim=DF*5+5+2+1
        n_class=2

        comb_data_X = []
        data_X = []
        data_Y = []

        model = load('./best_models/RF_model_2_best_new')

        if specific_bench in filepath and '_2_' not in filepath and '_3_' not  in filepath:
            print (filepath)
            # testing

            q2latchname={}

            add_q2latchname(benchname, filepath, q2latchname)

            file=filepath+f'\{benchname}_clean.bench'
            report_dir=filepath+f'/{benchname}_time_reports/'
            print (file)

            netlist_graph = ntk_parser(file)

            nxG = customgraph2networkx(netlist_graph)

            G_fflatch_only = fflatch_graph(nxG)

            # remove clk and reset node
            all_nodes = list(G_fflatch_only.nodes())

            for node in all_nodes:
                if node == 'rst':
                    G_fflatch_only.remove_node(node)
                if node == 'reset':
                    G_fflatch_only.remove_node(node)
                if node == 'clk':
                    G_fflatch_only.remove_node(node)

            test_nodes = []
            add_dataset(benchname, seq_sig, q2latchname, report_dir, DB, DF, G_fflatch_only, nxG, data_X, comb_data_X, data_Y, test_nodes)

            data_X = np.array(data_X)
            data_Y = np.array(data_Y)

            # normalization
            data_X = feature_normalization(data_X)
            print (data_X.shape)
            print (len(test_nodes))
            print (data_Y.shape)

            Y_test_label = np.argmax(data_Y, axis=1)


            #results = model.evaluate(X_test, Y_test, batch_size=32)
            y_pred=model.predict(data_X)
            test_acc = metrics.accuracy_score(Y_test_label, y_pred)

            print("Accuracy:", test_acc)
            print (confusion_matrix(Y_test_label, y_pred))

            all_results[x[0]]=all_results.get(x[0], [])+[test_acc]

            fbo=open(f"./all_results/{benchname}_result.txt", "w")
            fbo2 = open(f"./step1_output_all_find_LDs/{benchname}_LDs.txt", "w")
            for idx, node in enumerate(test_nodes):
                if y_pred[idx]==1:
                    tag='isLD'
                elif y_pred[idx]==0:
                    tag='notLD'

                if '_PO' in node:
                    node=node[:-3]

                if y_pred[idx]==1:
                    if idx != len(test_nodes) - 1:
                        fbo2.write(f'{q2latchname[node]}\n')
                    elif idx == len(test_nodes) - 1:
                        fbo2.write(f'{q2latchname[node]}')

                fbo.write(f'{q2latchname[node]}: {G_fflatch_only.nodes[test_nodes[idx]]["type"]} {tag}\n')
            fbo.close()
            fbo2.close()


    print (all_results)
    #return data_X, data_Y
    f = open(f"10datasets_cweights_simple_best_MLP_model_pytorch_2_testing_result_pytorch_output_result_new.txt", "w")
    for key, val in all_results.items():
        f.write(f'{key}:{val}\n')

    f.close()

seed(1)
benchpath="./dataset_seed1_test_wo_remove_LD"
all_bench='s'
seq_sig='reg' # signature of sequential logic, only re-elaborate can generate this signature

DB = 1  # depth for backward (towards inputs)
DF = 1  # depth for forward (towards outputs)


generate_dataset(all_bench, benchpath, DB, DF, seq_sig)









