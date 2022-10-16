import sys
import glob
import math
import os
import re, random
from Ntk_Struct_PO_cmu import *
from Ntk_Parser_PO_cmu import *
from fflatch_only_graph_PO import *
import numpy as np
import networkx as nx
import collections


import h5py
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.sampler import SubsetRandomSampler

from matplotlib.lines import Line2D
from networkx.algorithms import bipartite

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

def remove_DD_LD(G_latch_only):
    G_prime = copy.deepcopy(G_latch_only)

    all_nodes_before_add = list(G_prime.nodes())
    for node in all_nodes_before_add:
        #if G_prime.nodes[node]['type'] == 'LATCH_L0' or G_prime.nodes[node]['type'] == 'LATCH_L1' or G_prime.nodes[node]['type'] == 'PO' or G_prime.nodes[node]['type'] == 'IPT' or G_prime.nodes[node]['type'] == 'DFF':
        if G_prime.nodes[node]['type'] == 'LATCH_L0' or G_prime.nodes[node]['type'] == 'LATCH_L1':
            continue

        if G_prime.nodes[node]['type'] == 'LATCH_LD':
            G_prime.remove_node(node)
            continue

        #if G_prime.nodes[node]['type'] == 'LATCH_DD':
        in_nodes = []
        out_nodes = []
        for edge in G_prime.out_edges(node):
            first_node, second_node = edge
            out_nodes.append(second_node)
        for edge in G_prime.in_edges(node):
            first_node, second_node = edge
            in_nodes.append(first_node)
        for node_1 in in_nodes:
            for node_2 in out_nodes:
                #print(type(node_1))
                #print(node_1)
                G_prime.add_edge(node_1, node_2)
        G_prime.remove_node(node)

    return G_prime

def remove_LD_comb(nxG):
    nxG_wo_LD_comb = copy.deepcopy(nxG)

    LD_nodes = [x for x, y in nxG_wo_LD_comb.nodes(data=True) if y['type'] == 'LATCH_LD']

    remove_combs=set()
    for node in LD_nodes:
        #print ("Out:", nxG_wo_LD_comb.out_edges(node))
        for outedge in nxG_wo_LD_comb.out_edges(node):
            outnode=outedge[1]
            if 'LATCH' not in nxG_wo_LD_comb.nodes[outnode]['type']:
                #print (nxG_wo_LD_comb.nodes[outnode]['type'])
                remove_combs.add(outnode)

    #print (len(remove_combs), len(set(remove_combs)))
    for gate in remove_combs:
        nxG_wo_LD_comb.remove_node(gate)

    return nxG_wo_LD_comb

def coloring(G_MS_only, coloring_results):
    for node in G_MS_only:
        if G_MS_only.nodes[node]['type'] == 'LATCH_L0' or G_MS_only.nodes[node]['type'] == 'LATCH_L1':
            if node not in coloring_results:
                if not dfs_color(1, node, G_MS_only, coloring_results):
                    return False

    return True


def dfs_color(curcolor, node, G_MS_only, coloring_results):
    # exhaust every subgraph
    if node in coloring_results:
        if coloring_results[node]==curcolor:
            return True
        else:
            return False

    coloring_results[node]=curcolor

    for outedge in G_MS_only.out_edges(node):
        outnode=outedge[1]
        if 'LATCH' in G_MS_only.nodes[outnode]['type']:
            if not dfs_color(-curcolor, outnode, G_MS_only, coloring_results):
                return False

    for inedge in G_MS_only.in_edges(node):
        innode=inedge[0]
        if 'LATCH' in G_MS_only.nodes[innode]['type']:
            if not dfs_color(-curcolor, innode, G_MS_only, coloring_results):
                return False

    return True


def check_rules(node, G_MS_only):
    if G_MS_only.nodes[node]['type'] == 'LATCH_L0':

        flagMS = False
        for outedge in G_MS_only.out_edges(node):
            outnode = outedge[1]
            if G_MS_only.nodes[outnode]['type'] == 'LATCH_L0':  # M-M
                print ("M-M", node, outnode)


            if G_MS_only.nodes[outnode]['type'] == 'LATCH_L1':
                flagMS = True


    elif G_MS_only.nodes[node]['type'] == 'LATCH_L1':
        for outedge in G_MS_only.out_edges(node):
            outnode = outedge[1]
            if G_MS_only.nodes[outnode]['type'] == 'LATCH_L1':  # S-S
                print ("S-S", node, outnode)





def add_q2latchname(benchname, filepath, q2latchname):
    fi=open(filepath+'/'+benchname+'_latchname2Q_remove_LD','r')
    for line in fi:
        #print (line.rstrip().split(':'))
        q, latchname=line.rstrip().split(':')
        #print (q, latchname)
        q2latchname[q]=latchname
    fi.close()

def count_comp(G_MS_only):
    visited=set()

    numcomp=0
    for node in G_MS_only:
        if node not in visited:
            dfs(node, visited, G_MS_only)
            numcomp+=1

    return numcomp

def dfs(node, visited, G_MS_only):
    visited.add(node)

    for outedge in G_MS_only.out_edges(node):
        outnode=outedge[1]
        if outnode not in visited:
            dfs(outnode, visited, G_MS_only)

    for inedge in G_MS_only.in_edges(node):
        innode=inedge[0]
        if innode not in visited:
            dfs(innode, visited, G_MS_only)

def generate_dataset(specific_bench,benchpath,DB, DF, seq_sig):

    for idx, filepath in enumerate(glob.glob(benchpath + '/*')):
        print (filepath)
        x = re.findall(r"^\.\/dataset_seed1\\([A-Za-z0-9\_]+)$", filepath)
        #print (x[0])
        benchname = x[0].split('_')[0]

        print (benchname)

        if specific_bench in filepath:
            file=filepath+f'\{benchname}_clean_remove_LD.bench'
            report_dir=filepath+f'/{benchname}_time_reports_remove_LD/'
            #print (file)
            netlist_graph = ntk_parser(file)

            nxG = customgraph2networkx(netlist_graph)
            print (f"There are {nxG.number_of_edges()} edges in complete graph")

            G_latch_only = latch_graph(nxG)
            print (f"There are {G_latch_only.number_of_edges()} edges in latch graph")

            # remove clk and reset node
            all_nodes = list(G_latch_only.nodes())

            for node in all_nodes:
                if node == 'rst':
                    G_latch_only.remove_node(node)
                if node == 'reset':
                    G_latch_only.remove_node(node)
                if node == 'clk':
                    G_latch_only.remove_node(node)

            #for node in G_latch_only:
                #check_rules(node, G_latch_only)
            all_nodes = list(G_latch_only.nodes())

            fo = open(f"./output_L2PIPO/{benchname}_PIPO_remove_LD.txt", "w")
            all_gate_type=set()
            for node in all_nodes:
                PI=False
                PO=False
                if 'LATCH' in G_latch_only.nodes[node]['type']:
                    for inedge in G_latch_only.in_edges(node):
                        innode=inedge[0]
                        if G_latch_only.nodes[innode]['type']=="IPT" and 'key' not in str(innode):
                            PI=True

                            if G_latch_only.nodes[node]['type']=='LATCH_DD': # DD connect to PI
                                #print (node, innode)
                                for DDoutedge in G_latch_only.out_edges(node): # check fanout of PI-DD-
                                    DDoutnode=DDoutedge[1]
                                    #print (node,DDoutnode, G_latch_only.nodes[DDoutnode]['type'] )

                        if G_fflatch_only.nodes[innode]['type'] == "DFF":
                            PI = True


                    for outedge in G_latch_only.out_edges(node):
                        outnode=outedge[1]
                        if G_latch_only.nodes[outnode]['type'] == "PO":
                            PO=True

                        if G_fflatch_only.nodes[outnode]['type'] == "DFF":
                            print (f"Fanout DFF: {node},{G_fflatch_only.nodes[node]['type']} ")
                            PO = True

                    if PI and PO:
                        print (node,G_latch_only.nodes[node]['type'])

                    fo.write(f"{node} {PI} {PO} {G_latch_only.nodes[node]['type']}\n")


            fo.close()




benchpath="./dataset_seed1"

all_bench='s'
seq_sig='reg' # signature of sequential logic, only re-elaborate can generate this signature

DB = 1  # depth for backward (towards inputs)
DF = 1  # depth for forward (towards outputs)

generate_dataset(all_bench, benchpath, DB, DF, seq_sig)









