"""
Build control flow graphs from binaries. Makes use of the angr (docs.angr.io) binary analysis toolkit.
To get this to run I did:

conda create -n angr python=3
conda activate angr
pip install angr
conda install numpy, scipy, matplotlib, pyyaml

- Alex
"""
import angr
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
import collections, argparse, random, time, os, logging, datetime, signal
import yaml

def get_graph(bin_fileloc, attributes, plot=False, log=False, summarise=[]):
    """
    Takes binary files and returns the adjacency and feature matrix (for input to GNN) of the associated control flow graph (cfg).

    Nodes of the cfg are code blocks and directed edges represent control flow.
    """
    if not log: # add them if you see any other logs
        logging.getLogger('angr.analyses').setLevel('CRITICAL')
        logging.getLogger('angr.project').setLevel('CRITICAL')
        logging.getLogger('pyvex.lifting.libvex').setLevel('CRITICAL')
        logging.getLogger('angr.engines').setLevel('CRITICAL')

    p = angr.Project(bin_fileloc, load_options={'auto_load_libs': False})  

    try:
        cfg = p.analyses.CFGFast(force_complete_scan=False, show_progressbar=True) # fore_complete_scan=True does a linear disassembly after the recursive.
        G = cfg.graph
        # cfg.normalize() # not sure what this does and if its needed

    except TimeOut:
        raise

    except Exception as e:
        print('{} failed with exception {}'.format(os.path.basename(bin_fileloc), e))

        if summarise:
            summarise[0].append(((os.path.basename(bin_fileloc), str(e))))

        return 0, 0, False

    if len(G.nodes()) in [0, 1]:
        print('{} failed with recursive disassembly, attempting linear disassembly...'.format(os.path.basename(bin_fileloc)))

        try:
            cfg = p.analyses.CFGFast(force_complete_scan=True, show_progressbar=True)
            G = cfg.graph

        except TimeOut:
            raise

        except Exception as e:
            print('{} Linear disassembly failed with exception {}'.format(os.path.basename(bin_fileloc), e))

            if summarise:
                summarise[0].append(((os.path.basename(bin_fileloc), str(e))))

            return 0, 0, False

        if len(G.nodes()) in [0, 1]:
            print('Linear dissassemly failed, found {} node/s'.format(len(G.nodes())))

            if summarise:
                summarise[0].append((os.path.basename(bin_fileloc), '{} node/s'.format(len(G.nodes()))))

            return 0, 0, False

    if len(G.nodes()) > 10000: # A will be >100mb. GPU will prob run out of memory when trying to backprop.
        print('{} too large with {} nodes at {} bytes'.format(os.path.basename(bin_fileloc), len(G.nodes), os.path.getsize(bin_fileloc)))

        if summarise:
            summarise[1].append((os.path.basename(bin_fileloc), os.path.getsize(bin_fileloc)))    

        return 0, 0, False

    if plot:
        if len(G.nodes()) > 1000:
            print("Too large to plot")

        else:
            nx.draw(cfg.graph, node_size=70)
            plt.show()

    A = nx.to_numpy_array(G, dtype=bool)
    X = np.zeros((len(G.nodes()), len(attributes.keys()) + 1), dtype=np.uint16) # first column is size in bytes.
    category_columns = { category : i + 1 for i, category in enumerate(attributes.keys()) }

    for i, node in enumerate(list(G)):  
        try:
            X[i, 0] += node.block.size

            for insn in node.block.capstone.insns:
                opcode_line = str(insn).split('\t')[1].split() # Occasionly opcodes have prefixes

                for opcode in opcode_line:
                    found = False
                    for category, opcodes in attributes.items():
                        n = opcodes.count(opcode)

                        if n:
                            X[i,category_columns[category]] += n
                            found = True

                    if (not found) and summarise:
                        summarise[2][opcode] += 1

        except AttributeError: # block is an API call so node has no block
            pass

    return A, X, True


def main(input_dir, output_dir, opcode_categories, malicious, log, summary, test, plot):
    with open(opcode_categories, 'r') as f:
        opcodes = yaml.load(f, Loader=yaml.FullLoader)

    bin_filepaths = [ os.path.join(input_dir, filename) for filename in os.listdir(input_dir) ]

    if test:
        bin_filepaths = random.sample(bin_filepaths, 10)
    
    if summary:
        failed_cfgs = []
        bins_too_big = []
        missed_opcodes = collections.Counter()
        summary = (failed_cfgs, bins_too_big, missed_opcodes)

    start = time.time()
    signal.signal(signal.SIGALRM, handler)
    for i, bin_filepath in enumerate(bin_filepaths):

        try:
            signal.alarm(1200) # 20mins until timeout as CFGFast can take > 1 day (happens very rarely)
            A, X, success = get_graph(bin_filepath, opcodes, log=log, summarise=summary, plot=plot)
        
        except TimeOut:
            print('{} timeout. Moving on to next binary...'.format((os.path.basename(bin_filepath))))

            if summary:
                summary[0].append((os.path.basename(bin_filepath), "Timeout"))

            continue

        signal.alarm(0) # disable alarm

        if success:
            np.savez(os.path.join(output_dir, os.path.splitext(os.path.basename(bin_filepath))[0]) + '.npz', A=A, X=X, mal=np.array([malicious]))
            
            if test:
                plt.imshow(A)
                plt.colorbar()
                plt.show()

                plt.imshow(X, aspect='auto', interpolation='none')
                plt.colorbar()  
                plt.show()

        if ((i + 1) % 20) == 0:
            end = time.time()
            print('[{}/{}] - ({:2.2%}) - {:.0f}s'.format(i+1, len(bin_filepaths), float(i + 1)/float(len(bin_filepaths)), end - start))

    end = time.time()

    if summary:
        now = datetime.datetime.now().strftime("_%d%m%Y-%H%M%S")

        with open(os.path.join(output_dir, 'summary') + now + '.txt', 'w') as f:
            f.write('Binaries analysed: {}\n\nTime elapsed: {:.0f}s\n\nFailed CFGs: {}\n{}\n\nBinaries too large: {}\n{}\n\nUncategorized opcodes:\n{}\n'.format(
                    len(bin_filepaths), end - start, len(summary[0]), str(summary[0]), len(summary[1]), str(summary[1]),str(summary[2])))


def handler(signum, frame):
    raise TimeOut("Timeout")


class TimeOut(Exception):
    pass


def parse_arguments():
    parser = argparse.ArgumentParser(description="Create .npz file with adjacency matrix, feature matrix and binary indicator with 1 for malware")
    parser.add_argument("input_dir", help="Directory of binaries")
    parser.add_argument("output_dir", help="Directory to output cfg files")
    parser.add_argument("opcode_categories", help="yaml file with opcode categories for feature matrix")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--benign", action="store_true", help="Binaries are benign")
    group.add_argument("--malware", action="store_true", help="Binaries are malicious")
    parser.add_argument("-l", "--log", action="store_true",
                        help="Display CFGFast log (constantly dump errors and warnings to your terminal)")
    parser.add_argument("-s", "--summary", action="store_true",
                        help="Leave a summary file with failed cfgs, missed opcode counter, and binaries too big for a cfg")
    parser.add_argument("-t", "--test", action="store_true",
                        help="Extract 10 random binaries from the input_dir and plot the adjacency and feature matrix")
    parser.add_argument("-p", "--plot", action="store_true",
                        help="Plot cfg if there are not too many (>1000) nodes")
    args = parser.parse_args()
    
    return (args.input_dir, args.output_dir, args.opcode_categories, args.malware, args.log, args.summary, args.test, args.plot)


if __name__ == "__main__":
    arguments = parse_arguments()
    main(*arguments)
