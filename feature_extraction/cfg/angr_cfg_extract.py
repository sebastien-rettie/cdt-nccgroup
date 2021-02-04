"""
Build graphs from binaries. Makes use of the angr (docs.angr.io) binary analysis toolkit. Looks like a useful tool but there is a lot 
about it I still don't understand.
Test it out with the test binaries. Two of them are windows drivers and the other one is from the C++ class.
To get this to run I did:

conda create --n angr python=3
conda activate angr
pip install angr
conda install numpy
conda install scipy (think networkx needs this to draw the graph)
conda install matplotlib

- Alex
"""

import angr
import sys
import networkx as nx
import numpy as np
from io import StringIO
from matplotlib import pyplot as plt

def get_graph(bin_fileloc, force_complete_scan=False, plot=False):
    """
    Takes binary files and returns the adjacency and feature matrix (for input to GNN) of the associated control flow graph (cfg).

    Nodes of the cfg are code blocks and directed edges represent control flow. 

    Currently the features for the code blocks are: # of instructions, # of nop, # of call, # of jmp.
    """
    p = angr.Project(bin_fileloc, load_options={'auto_load_libs': False})
    cfg = p.analyses.CFGFast(force_complete_scan=force_complete_scan)  
    # cfg.normalize()
    G = cfg.graph

    A = nx.to_numpy_array(G, dtype=np.bool)
    X = np.zeros((len(G.nodes()), 4), dtype=np.int8)

    if plot:
        nx.draw(cfg.graph, node_size=70)
        plt.show()

    for i, node in enumerate(list(G)):
        try:
            old_stdout = sys.stdout
            sys.stdout = mystdout = StringIO() # this capture the print output to mystdout, a janky way to get the assembly code as a str. 

            print(node.block.capstone)

            sys.stdout = old_stdout

            code_block = mystdout.getvalue()

            X[i,1] = code_block.count('\tnop\t')
            X[i,2] = code_block.count('\tcall\t')
            X[i,3] = code_block.count('\tjmp\t')

            X[i,0] = node.block.instructions

        except AttributeError: # Sometimes the code blocks don't exist, think these are calls to external functions. Need to ask NCC guys.
            sys.stdout = old_stdout

    return A, X


def main():
    # A, X = get_graph('test_benign_binaries/B411BC77CC5097E765D9DC9E215F56797347EAD23A6613EA90A9BE296E83E42E00.blob')
    A, X = get_graph('test_benign_binaries/5E098569FBCA0228E83966E8A74F0EC0E3BF69EAC8228D1EB122A44D68A0A0A800.blob')
    # A, X = get_graph('test_benign_binaries/hi.cpp.o')

    plt.imshow(A)
    plt.colorbar()
    plt.show()

    plt.imshow(X, aspect='auto', interpolation='none')
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    main()
