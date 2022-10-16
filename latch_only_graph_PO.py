# Executed in Python 3.6
import re
from collections import OrderedDict
from Ntk_Struct_PO_cmu import *
import sys
import networkx as nx
import matplotlib.pyplot as plt
import numpy
import copy


def latch_graph(G):
    G_prime = copy.deepcopy(G)
    all_nodes_before_add=list(G_prime.nodes())
    for node in all_nodes_before_add:
        #print (node, G_prime.nodes[node])
        #if (G_prime.nodes[node]['type'] == 'LATCH_L0' or G_prime.nodes[node]['type'] == 'LATCH_L1' or G_prime.nodes[node]['type'] == 'LATCH_LD' or G_prime.nodes[node]['type'] == 'LATCH_DD' or G_prime.nodes[node]['type'] == 'DFF' or G_prime.nodes[node]['type'] == 'DFFS' or G_prime.nodes[node]['type'] == 'DFFR' or G_prime.nodes[node]['type'] == 'DFFRS' or G_prime.nodes[node]['type'] == 'State_FF'):
        if (G_prime.nodes[node]['type'] == 'LATCH_L0' or G_prime.nodes[node]['type'] == 'LATCH_L1' or G_prime.nodes[node]['type'] == 'LATCH_LD' or G_prime.nodes[node]['type'] == 'LATCH_DD'):
            if "_PO" in node: # the DFF node is an PO, add an extra PO node
                #print (node)
                add_POnode=node+"_POnode"
                G_prime.add_node(add_POnode, type='PO')
                G_prime.add_edge(node, add_POnode)
            continue

        if G_prime.nodes[node]['type']=='IPT':
            continue

        if "_PO" in node: # a comb gate is PO, add an extra PO node
            add_POnode = node + "_POnode"
            G_prime.add_node(add_POnode, type='PO')
            G_prime.add_edge(node, add_POnode)

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

    
if __name__ == '__main__':
    start_number = -1
    for i in range(1,len(sys.argv)):
        filename = sys.argv[i]
        pf = filename.split("/")[-1].split(".")[0]
        print(pf)
        Graph = nx.MultiDiGraph()
        Graph1 = ff_graph(Graph, pf)


