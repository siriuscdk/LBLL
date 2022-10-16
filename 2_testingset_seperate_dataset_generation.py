import sys
import glob
import math
import re, random
from Ntk_Struct_PO_cmu import *
from Ntk_Parser_PO_cmu import *
from fflatch_only_graph_PO import *
import numpy as np
import networkx as nx
import collection
import h5py
from sklearn.preprocessing import MinMaxScaler


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
    fi=open(filepath+'/'+benchname+'_latchname2Q_remove_LD','r')
    for line in fi:

        q, latchname=line.rstrip().split(':')
        q2latchname[q]=latchname
    fi.close()

def add_delay(node, benchname, report_dir, q2latchname, seq_sig):
    latchname=q2latchname[node][:-7]
    PIflag=False # this latch connects to input

    f1=open(report_dir+latchname+'.from','r')
    f2 = open(report_dir + latchname + '.to', 'r')
    for line in f1:

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



def add_dataset(benchname, seq_sig, q2latchname,report_dir, DB, DF, G_fflatch_only, nxG, data_X, comb_data_X, data_Y):
    for node in G_fflatch_only:
        if not (G_fflatch_only.nodes[node]['type'] == 'LATCH_L0' or G_fflatch_only.nodes[node]['type'] == 'LATCH_L1' or G_fflatch_only.nodes[node]['type'] == 'LATCH_LD' or G_fflatch_only.nodes[node]['type'] == 'LATCH_DD'):
           continue

        if len(G_fflatch_only.in_edges(node))==0:
            continue

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
            #fraction+=len(curFO&parentFO)
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

        vector = backl + forwl + [fraction / tot_num_FI, Sfraction/tot_num_FI, Sfraction2/tot_num_FO,latchloop]

        if len(G_fflatch_only.out_edges(node))==1 or len(G_fflatch_only.in_edges(node))==1:
            vector.append(1)
        else:
            vector.append(0)


        if '_PO' in node:
            nodename=node[:-3]
            #print (nodename)
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



        #print (vector)
        data_X.append(vector)

        # add labels according to the latch's type
        if G_fflatch_only.nodes[node]['type'] == 'LATCH_L0':
            data_Y.append(np.array([1, 0]))
        elif G_fflatch_only.nodes[node]['type'] == 'LATCH_L1':
            data_Y.append(np.array([1, 0]))
        elif G_fflatch_only.nodes[node]['type'] == 'LATCH_DD':
            data_Y.append(np.array([0, 1]))
        elif G_fflatch_only.nodes[node]['type'] == 'LATCH_LD':
            data_Y.append(np.array([0, 1]))


def generate_dataset(specific_bench, datasetnum, benchpath,DB, DF, seq_sig):
    for idx, filepath in enumerate(glob.glob(benchpath + '/*')):
        print (filepath, datasetnum)
        regfilepath=rf"^\.\/dataset_seed{str(datasetnum)}\\([A-Za-z0-9\_]+)$"
        x = re.findall(regfilepath, filepath)
        #print (x)
        benchname=x[0].split('_')[0]

        data_X = []
        data_Y = []
        comb_data_X = []
        if specific_bench in filepath:
            q2latchname={}

            add_q2latchname(benchname, filepath, q2latchname)

            #print (q2latchname)


            file=filepath+f'\{benchname}_clean_remove_LD.bench'
            report_dir=filepath+f'/{benchname}_time_reports_remove_LD/'
            print (file)

            netlist_graph = ntk_parser(file)

            nxG = customgraph2networkx(netlist_graph)

            G_fflatch_only = fflatch_graph(nxG)

            # remove clk and reset node
            all_nodes = list(G_fflatch_only.nodes())
            #print (all_nodes)

            for node in all_nodes:
                if node == 'rst':
                    G_fflatch_only.remove_node(node)
                if node == 'reset':
                    G_fflatch_only.remove_node(node)
                if node == 'clk':
                    G_fflatch_only.remove_node(node)

            add_dataset(benchname, seq_sig, q2latchname,report_dir, DB, DF, G_fflatch_only, nxG, data_X,comb_data_X, data_Y)

            data_X = np.array(data_X)
            data_Y = np.array(data_Y)
            comb_data_X = np.array(comb_data_X) # comb gates info

            # normalization
            data_X = feature_normalization(data_X)
            print (data_X.shape)
            print (data_Y.shape)

            #print (data_X)

            with h5py.File(f"./all_testing_sets_2class/{x[0]}_dataset.hdf5", 'w') as hf:
                hf.create_dataset('X_train', data=data_X)
                hf.create_dataset('Y_train', data=data_Y)


def generate_all_datasets(datasets):
    all_bench = 's'
    seq_sig = 'reg'  # signature of sequential logic, only re-elaborate can generate this signature

    DB = 1  # depth for backward (towards inputs)
    DF = 1  # depth for forward (towards outputs)
    # count_visited_FF = True

    for num in datasets:
        benchpath="./dataset_seed"+str(num)
        generate_dataset(all_bench, num, benchpath, DB, DF, seq_sig)

datasets=[1]
generate_all_datasets(datasets)










