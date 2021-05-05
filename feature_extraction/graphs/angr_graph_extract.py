"""
Build control flow graphs and callgraphs from binaries. Makes use of the angr (docs.angr.io) binary analysis toolkit.
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
import collections, argparse, random, time, os, logging, datetime, queue, multiprocessing
import yaml

def get_graph(bin_fileloc, cfg_attributes={}, opcode_cols={}, cg_attributes={}, all_api_names=[], 
              plot=False, log=False, cfg_summarise=[], cg_summarise=[]):
    """
    Takes binary files and returns the adjacency and feature matrix (for input to GNN) of the associated cfg or cg.
    """
    if not log: # add them if you see any other logs
        logging.getLogger('angr.analyses').setLevel('CRITICAL')
        logging.getLogger('angr.project').setLevel('CRITICAL')
        logging.getLogger('pyvex.lifting.libvex').setLevel('CRITICAL')
        logging.getLogger('angr.engines').setLevel('CRITICAL')

    if cfg_attributes and cg_attributes:
        cfg_good, cg_good = True, True

    elif cfg_attributes:
        cfg_good, cg_good = True, False

    elif cg_attributes:
        cfg_good, cg_good = False, True

    try:
        p = angr.Project(bin_fileloc, load_options={'auto_load_libs': False})  

    except Exception as e:
        print('{} failed with exception {}'.format(os.path.basename(bin_fileloc), e))

        if cfg_summarise:
            cfg_summarise[0].append(((os.path.basename(bin_fileloc), str(e))))
            cg_summarise[0].append(((os.path.basename(bin_fileloc), str(e))))

        return 0, 0, 0, False, 0, 0, False

    try:
        cfg = timed_cfgfast(p, False)
        # cfg.normalize() # not sure what this does and if its needed
        G = cfg.graph
        G_func = cfg.functions.callgraph

    except TimeOut:
        raise

    except Exception as e:
        print('{} failed with exception {}'.format(os.path.basename(bin_fileloc), e))

        if cfg_summarise:
            cfg_summarise[0].append(((os.path.basename(bin_fileloc), str(e))))
            cg_summarise[0].append(((os.path.basename(bin_fileloc), str(e))))

        return 0, 0, 0, False, 0, 0, False

    if len(G.nodes()) in [0, 1]:
        print('{} failed with recursive disassembly, attempting linear disassembly...'.format(
            os.path.basename(bin_fileloc)))

        try:
            cfg = timed_cfgfast(p, True)
            G = cfg.graph
            G_func = cfg.functions.callgraph

        except TimeOut:
            raise

        except Exception as e:
            print('{} Linear disassembly failed with exception {}'.format(
                os.path.basename(bin_fileloc), e
                ))

            if cfg_summarise:
                cfg_summarise[0].append(((os.path.basename(bin_fileloc), str(e))))
                cg_summarise[0].append(((os.path.basename(bin_fileloc), str(e))))

            return 0, 0, 0, False, 0, 0, False

        if len(G.nodes()) in [0, 1]:
            print('Linear dissassemly failed, found {} cfg node/s'.format(len(G.nodes())))

            if cfg_summarise:
                cfg_summarise[0].append((os.path.basename(bin_fileloc), '{} cfg node/s'.format(len(G.nodes()))))
                cg_summarise[0].append((os.path.basename(bin_fileloc), '{} cfg node/s'.format(len(G.nodes()))))

            return 0, 0, 0, False, 0, 0, False

    if (len(G_func.nodes()) in [0,1]) and cg_good:
        print('Linear dissassemly failed for callgraph, found {} cg node/s')

        if cfg_summarise:
            cg_summarise[0].append((os.path.basename(bin_fileloc), '{} cg node/s'.format(len(G_func.nodes()))))

        cg_good = False 

    if (len(G_func.nodes()) > 10000) and cg_good: # Batchsize will need to account for largest samples so putting a limit for convenience
        print('{} cg too large with {} nodes at {} bytes'.format(
            os.path.basename(bin_fileloc), len(G.nodes), os.path.getsize(bin_fileloc)))

        if cfg_summarise:
            cfg_summarise[1].append((os.path.basename(bin_fileloc), os.path.getsize(bin_fileloc)))    
            cg_summarise[1].append((os.path.basename(bin_fileloc), os.path.getsize(bin_fileloc)))  

        return 0, 0, 0, False, 0, 0, False # cfg is always larger than cg so can return here

    if (len(G.nodes()) > 10000) and cfg_good:
        print('{} cfg too large with {} nodes at {} bytes'.format(
            os.path.basename(bin_fileloc), len(G.nodes), os.path.getsize(bin_fileloc)
            ))

        if cfg_summarise:
            cfg_summarise[1].append((os.path.basename(bin_fileloc), os.path.getsize(bin_fileloc)))    

        if not cg_good:
            return 0, 0, 0, False, 0, 0, False

        else:
            cfg_good = False

    if plot:
        # temporary
        # sub_G = nx.MultiDiGraph()
        # def create_subgraph(G, sub_G, start_node):
        #     for n in G.successors(start_node):
        #         sub_G.add_edge(start_node, n)
        #         create_subgraph(G, sub_G, n)

        # create_subgraph(G_func, sub_G, 16785136)
        # nx.draw(sub_G, node_size=210, with_labels=True)
        # plt.show()
        if len(G.nodes()) > 1000:
            print("cfg too large to plot")

        elif cfg_good:
            nx.draw(G, node_size=70)
            plt.title('CFG')
            plt.show()

        if len(G_func.nodes()) > 1000:
            print("cg too large to plot")

        elif cg_good:
            cmap = []
            for addr in G_func.nodes():
                func = cfg.functions.function(addr)

                if func.name == '_start':
                    cmap.append('green')

                elif func.name in all_api_names:
                    cmap.append('red')

                elif func.name.split('_')[0] == 'sub':
                    cmap.append('blue')

                else:
                    cmap.append('black')

            nx.draw(G_func, node_size=140, node_color=cmap)
            plt.title('CG')
            plt.show()
    
    if cfg_good:
        C = 0
        A = nx.to_numpy_array(G, dtype=np.uint8)

        if cfg_attributes != "none":
            X = np.zeros((len(G.nodes()), len(cfg_attributes.keys()) + 1), dtype=np.uint16) # first column is size in bytes.
            cfg_category_columns = { category : i + 1 for i, category in enumerate(cfg_attributes.keys()) }

            for i, node in enumerate(G.nodes()):  
                try:
                    X[i, 0] += node.block.size

                    for insn in node.block.capstone.insns:
                        opcode_line = str(insn).split('\t')[1].split() # Occasionly opcodes have prefixes

                        for opcode in opcode_line:
                            found = False
                            for category, opcodes in cfg_attributes.items():
                                n = opcodes.count(opcode)

                                if n:
                                    X[i,cfg_category_columns[category]] += n
                                    found = True

                            if (not found) and cfg_summarise:
                                cfg_summarise[2][opcode] += 1

                except AttributeError: # block is an API call so node has no block
                    pass

        else:
            for i, node in enumerate(G.nodes()):  
                try:
                    for insn in node.block.capstone.insns:
                        opcode_line = str(insn).split('\t')[1].split() # Occasionly opcodes have prefixes

                        for opcode in opcode_line:
                            if opcode not in opcode_cols.keys():
                                opcode_cols[opcode] = max(opcode_cols.values()) + 1


                except AttributeError: # node is an API call so has no block
                    pass
            
            C = np.array(sorted(opcode_cols, key=opcode_cols.get), dtype='object')
            X = np.zeros((len(G.nodes()), len(opcode_cols.keys())), dtype=np.uint16)

            for i, node in enumerate(G.nodes()):  
                try:
                    X[i, 0] += node.block.size

                    for insn in node.block.capstone.insns:
                        opcode_line = str(insn).split('\t')[1].split() # Occasionly opcodes have prefixes

                        for opcode in opcode_line:
                            X[i, opcode_cols[opcode]] += 1

                except AttributeError: # block is an API call so node has no block
                    pass
            

    if cg_good:
        A_func = nx.to_numpy_array(G_func, dtype=np.uint8)
        X_func = np.zeros((len(G_func.nodes()), len(cg_attributes.keys())), dtype=np.uint16) 
        cg_category_columns = { category : i for i, category in enumerate(cg_attributes.keys()) }

        attr_mapping = { addr : cfg.functions.function(addr).name for addr in G_func.nodes() }
        nx.set_node_attributes(G_func, attr_mapping, "func_name")

        api_nodes = { 
            node : data["func_name"] for node, data in G_func.nodes(data=True) if data["func_name"] in all_api_names }
        seen_categories = [ 
            category for category, api_lst in cg_attributes.items() if set(api_lst).intersection(set(api_nodes.values())) ]

        if (len(api_nodes.keys()) == 0):
            cg_good = False
            if cfg_summarise:
                cg_summarise[2].append((
                    os.path.basename(bin_fileloc), 
                    "Of {} nodes, none were recognised as categorised api calls".format(len(G_func.nodes))))
    
        for i, (node, data) in enumerate(G_func.nodes(data=True)):
            for api_node, api_name in api_nodes.items():
                try:
                    path_length = len(nx.shortest_path(G_func, source=node, target=api_node))

                except nx.exception.NetworkXNoPath:
                    continue

                for category in seen_categories:
                    if (api_name in cg_attributes[category]):
                        if (X_func[i, cg_category_columns[category]] == 0) or (X_func[i, cg_category_columns[category]] > path_length): # looking for the shortest path to this API type
                            X_func[i, cg_category_columns[category]] = path_length

            if cfg_summarise:
                if data["func_name"] not in api_nodes.values():
                    if (not data["func_name"].startswith('sub')):
                        cg_summarise[3][data["func_name"]] += 1
            
    if (cg_good and cfg_good):
        return A, X, C, True, A_func, X_func, True 

    elif cfg_good:
        return A, X, C, True, 0, 0, False

    elif cg_good:
        return 0, 0, 0, False, A_func, X_func, True
    
    else:
        return 0, 0, 0, False, 0, 0, False


def timed_cfgfast(project, force_complete_scan):
    Q = multiprocessing.Queue()
    kwargs = { "force_complete_scan" : force_complete_scan, "show_progressbar" : True}
    p = multiprocessing.Process(target=out_to_queue, args=(project.analyses.CFGFast, (), kwargs, Q))
    p.start()
    
    try:
        cfg = Q.get(timeout=300) # If CFGFast takes more than 5 minutes its probably going to take hours (or the process is dead)

        if type(cfg) == str:
            raise ProcessDead('ProcessDead')

        else:
            return cfg
    
    except queue.Empty:
        p.terminate()

        raise TimeOut

    except ProcessDead:
        raise


def out_to_queue(func, args, kwargs, Q):

    try:
        Q.put(func(*args, **kwargs))

    except Exception as e:
        Q.put(str(e))


def main(input_dir, output_dir, opcode_categories, api_categories, malicious, log, summary, test_n,
         plot, target_binary, starting_index):

    opcodes, opcode_cols = {}, {}
    if opcode_categories == "none":
        opcodes = "none"
        opcode_cols = { "block_bytes" : 0 }

    elif opcode_categories:
        with open(opcode_categories, 'r') as f:
            opcodes = yaml.load(f, Loader=yaml.FullLoader)

    apis, all_apis = [], {}
    if api_categories:
        with open(api_categories, 'r') as f:
            apis = yaml.load(f, Loader=yaml.FullLoader)
        all_apis = [ api_name for api_name_lst in apis.values() for api_name in api_name_lst ] 
        
    bin_filepaths = [ os.path.join(input_dir, filename) for filename in os.listdir(input_dir) ]

    if test_n:
        bin_filepaths = random.sample(bin_filepaths, test_n)
    
    if summary:
        cfg_summary = ([], [], collections.Counter())
        cg_summary = ([], [], [], collections.Counter())
    else:
        cfg_summary = ()
        cg_summary = ()

    start = time.time()
    for i, bin_filepath in enumerate(bin_filepaths):
        if i < starting_index:
            continue

        if target_binary:
            if os.path.basename(bin_filepath) != target_binary:
                continue

        try:
            A, X, C, cfg_success, A_func, X_func, cg_success = get_graph(
                bin_filepath, cfg_attributes=opcodes, opcode_cols=opcode_cols, cg_attributes=apis,
                all_api_names=all_apis, log=log, plot=plot, cfg_summarise=cfg_summary, cg_summarise=cg_summary)

        except TimeOut:
            print("{} timeout. Moving on to next binary...".format((os.path.basename(bin_filepath))))

            if summary:
                cfg_summary[0].append((os.path.basename(bin_filepath), "Timeout"))
                cg_summary[0].append((os.path.basename(bin_filepath), "Timeout"))

            continue

        if cfg_success:
            save_loc = os.path.join(output_dir, 'cfg')
            if not os.path.exists(save_loc):
                os.makedirs(save_loc)

            if opcode_categories == 'none':
                np.savez(
                    os.path.join(save_loc, os.path.splitext(os.path.basename(bin_filepath))[0] + '.npz'),
                    A=A, X=X, C=C, mal=np.array([malicious]))
            else:
                np.savez(
                    os.path.join(save_loc, os.path.splitext(os.path.basename(bin_filepath))[0] + '.npz'),
                    A=A, X=X, mal=np.array([malicious]))
            
            if test_n:
                print(os.path.basename(bin_filepath))

                plt.imshow(A, interpolation='none')
                plt.title('CFG Adjacency Matrix')
                plt.colorbar()
                plt.show()

                plt.imshow(X, aspect='auto', interpolation='none')
                plt.title('Feature Matrix - Number of Opcodes by Category')
                plt.colorbar()  
                plt.show()

        if cg_success:
            save_loc = os.path.join(output_dir, 'cg')
            if not os.path.exists(save_loc):
                os.makedirs(save_loc)

            np.savez(
                os.path.join(save_loc, os.path.splitext(os.path.basename(bin_filepath))[0] + '.npz'),
                 A=A_func, X=X_func, mal=np.array([malicious]))
            
            if test_n:
                print(os.path.basename(bin_filepath))

                plt.imshow(A_func, interpolation='none')
                plt.title('CG Adjacency Matrix')
                plt.colorbar()
                plt.show()

                plt.imshow(X_func, aspect='auto', interpolation='none')
                plt.title('Feature Matrix - Shortest Path to API call by Category')
                plt.colorbar()  
                plt.show()

        if ((i + 1) % 20) == 0:
            end = time.time()
            print('[{}/{}] - ({:2.2%}) - {:.0f}s'.format(
                i+1, len(bin_filepaths),
                float(i + 1 - starting_index)/float(len(bin_filepaths) - starting_index), end - start))

    end = time.time()

    now = datetime.datetime.now().strftime("_%d%m%Y-%H%M%S")

    if opcode_cols:
        save_loc = os.path.join(output_dir, 'cfg')
        with open(os.path.join(save_loc, 'opcodecols') + now + '.yaml', 'w') as f:
            yaml.dump(opcode_cols, f)

    if summary:
        if opcode_categories:
            save_loc = os.path.join(output_dir, 'cfg')
            with open(os.path.join(save_loc, 'summary') + now + '.txt', 'w') as f:
                f.write('Binaries analysed: {}\n\nTime elapsed: {:.0f}s\n\nFailed CFGs: {}\n{}\n\nBinaries too large: {}\n{}\n\nUncategorized opcodes:\n{}\n'.format(
                    len(bin_filepaths) - starting_index, end - start, len(cfg_summary[0]), 
                    str(cfg_summary[0]), len(cfg_summary[1]), str(cfg_summary[1]),str(cfg_summary[2])))

        if api_categories:
            save_loc = os.path.join(output_dir, 'cg')
            with open(os.path.join(save_loc, 'summary') + now + '.txt', 'w') as f:
                f.write('Binaries analysed: {}\n\nTime elapsed: {:.0f}s\n\nFailed CGs: {}\n{}\n\nBinaries too large: {}\n{}\n\nCGs without any API_names identified: {}\n{}\n\nUncategorized APIs:\n{}\n'.format(
                    len(bin_filepaths) - starting_index, end - start, len(cg_summary[0]),
                    str(cg_summary[0]), len(cg_summary[1]), str(cg_summary[1]), len(cg_summary[2]),
                    str(cg_summary[2]) ,str(cg_summary[3])))


class TimeOut(Exception): pass


class ProcessDead(Exception) : pass


def parse_arguments():
    parser = argparse.ArgumentParser(description="For each binary, create .npz file with adjacency matrix, feature matrix and \
                                    a binary indicator (1 if malware).Extracting both cfg and cg takes almost the same time \
                                    as extracting just one.")
    parser.add_argument("input_dir", help="Directory of binaries")
    parser.add_argument("output_dir", help="Directory to output cfg files")

    group1 = parser.add_mutually_exclusive_group()
    group1.add_argument("--benign", action="store_true", help="Binaries are benign")
    group1.add_argument("--malware", action="store_true", help="Binaries are malicious")

    parser.add_argument("--cfg", nargs='?', type=str, action="store", dest="OPCODE_CATEGORIES",
                        help="yaml file with opcode categories for feature matrix. If 'none' will give each unique opcode a column \
                              and save the column opcode names as a 1d array of strings in the npz. This will create feature matrices of increasing size as new \
                              opcodes are seen so you will need to go over the NPZs and resize the matrices according to desired \
                              opcodes.", default='')

    parser.add_argument("--cg", nargs='?', type=str, action="store", dest="API_CATEGORIES",
                        help="yaml file with opcode categories for feature matrix", default='')

    parser.add_argument("-l", "--log", action="store_true",
                        help="Display CFGFast log (constantly dump errors and warnings to your terminal)")
    parser.add_argument("-s", "--summary", action="store_true",
                        help="Leave a summary file with failed cfgs, missed opcode counter, and binaries too big for a cfg")
    parser.add_argument("-t", "--test", nargs='?', type=int, action="store", dest="N", default=0,
                        help="Extract N random binaries from the input_dir and plot the adjacency and feature matrix")
    parser.add_argument("-p", "--plot", action="store_true",
                        help="Plot cfg if there are not too many (>1000) nodes")
    parser.add_argument("--single", nargs='?', type=str, action="store", dest="TARGET_BINARY", default='')
    parser.add_argument("-i", "--starting_index", nargs='?', type=int, action="store", dest="STARTING_INDEX", default=0)

    args = parser.parse_args()

    if not (args.benign or args.malware):
        parser.error("Specify if binaries are benign or malware with --benign or --malware")

    if not (args.OPCODE_CATEGORIES or args.API_CATEGORIES):
        parser.error("Specify at least one of --cfg or --cg")

    return (args.input_dir, args.output_dir, args.OPCODE_CATEGORIES, args.API_CATEGORIES, args.malware,
            args.log, args.summary, args.N, args.plot, args.TARGET_BINARY, args.STARTING_INDEX)


if __name__ == "__main__":
    arguments = parse_arguments()
    main(*arguments)
