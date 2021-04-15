import argparse, os, csv, collections
import yaml
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt


def main(input_dir, output_dir):

    total_files = len(os.listdir(input_dir))
    opcodes = collections.Counter()

    for idx, filename in enumerate(os.listdir(input_dir)):
        if filename.endswith('.npz'):      
            data = np.load(os.path.join(input_dir, filename), allow_pickle=True)
            X, C = data['X'], data['C']

            for col_idx, opcode_col in enumerate(X.T):
                opcodes[C[col_idx]] += opcode_col.sum()

            if ((idx + 1) % 200) == 0:
                print("Counting opcodes [{}/{}] - ({:2.2%})".format(idx + 1, total_files, float(idx + 1)/float(total_files)))
                
    opcode_cols = { str(pair[0]) : [int(pair[1]), idx] for idx, pair in enumerate(opcodes.most_common(303)) }
    with open(os.path.join(output_dir, "opcode_cols.yaml"), 'w') as f:
        yaml.dump(opcode_cols, f)

    final_opcodes = [ str(pair[0]) for pair in opcodes.most_common(303) ]
    
    for idx, filename in enumerate(os.listdir(input_dir)):
        if filename.endswith('.npz'):      
            data = np.load(os.path.join(input_dir, filename), allow_pickle=True)
            A, X, C, mal = data['A'], data['X'], data['C'], data['mal']

            reduced_X = np.zeros((X.shape[0], 303))
            for opcode_idx, opcode in enumerate(C):
                if opcode in final_opcodes:
                    reduced_X.T[final_opcodes.index(opcode)] = X.T[opcode_idx]

            G = nx.from_numpy_matrix(np.matrix(A), create_using=nx.MultiDiGraph)

            nx.write_edgelist(G, "temp", data=False, delimiter=',')
            edge_list = [[],[]]

            with open("temp", 'r') as f:
                reader = csv.reader(f, delimiter=',')
                for row in reader:
                    edge_list[0].append(int(row[0]))
                    edge_list[1].append(int(row[1]))

            E = np.array(edge_list)

            np.savez(os.path.join(output_dir, filename), A=A, E=E, X=reduced_X, mal=mal)

            os.remove("temp")

            if ((idx + 1) % 200) == 0:
                print("Processing [{}/{}] - ({:2.2%})".format(idx + 1, total_files, float(idx + 1)/float(total_files)))
    

def parse_arguments():

    parser = argparse.ArgumentParser(
        description="Resize opcode feature matrices to use 303 most freqent opcodes only")

    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    args = parser.parse_args()
    
    return (args.input_dir, args.output_dir)


if __name__ == "__main__":
    arguments = parse_arguments()
    main(*arguments)