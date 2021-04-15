import argparse, os, csv
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt


def main(input_dir, output_dir):

    total_files = len(os.listdir(input_dir))
    for idx, filename in enumerate(os.listdir(input_dir)):
        if idx < 14200: continue

        if filename.endswith('.npz'):      
            if filename == "VirusShare_67729a3d206fa8f8f036f298ad56f11c.npz": continue # corrupted
            
            data = np.load(os.path.join(input_dir, filename))

            G = nx.from_numpy_matrix(np.matrix(data["A"]), create_using=nx.MultiDiGraph)

            nx.write_edgelist(G, 'temp', data=False, delimiter=',')
            edge_list = [[],[]]

            with open('temp', 'r') as f:
                reader = csv.reader(f, delimiter=',')
                for row in reader:
                    edge_list[0].append(int(row[0]))
                    edge_list[1].append(int(row[1]))

            E = np.array(edge_list)

            np.savez(os.path.join(output_dir, filename), A=data['A'], E=E, X=data['X'], mal=data['mal'])

            os.remove('temp')

            if ((idx + 1) % 100) == 0:
                print('[{}/{}] - ({:2.2%})'.format(idx + 1, total_files, float(idx + 1)/float(total_files)))


def parse_arguments():

    parser = argparse.ArgumentParser(description="Transform graph data into (edge_list, feature matrix, class)")
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")

    args = parser.parse_args()
    
    return (args.input_dir, args.output_dir)


if __name__ == "__main__":
    arguments = parse_arguments()
    main(*arguments)


    # ---------- plotting stuff ----------
    # odbcad32 is a nice looking cg
    # data = np.load('/unix/cdtdisncc/2021/cfg/classifier/data/cg/odbcad32.npz')

    # print(data["A"].shape)
    # print(data["X"].shape)
    # print(data["mal"])
    # for row in data["X"]:
    #     print(row)

    # G = nx.from_numpy_matrix(np.matrix(data["A"]), create_using=nx.MultiDiGraph)

    # attr_mapping = { n : data["X"][n] for n in G.nodes() }
    # nx.set_node_attributes(G, attr_mapping, "feature_vector")

    # cmap = []
    # for n in G.nodes:
    #     if any( G.nodes[n]["feature_vector"] == 1):
    #         cmap.append('red')
    #     else:
    #         cmap.append('blue')
    # nx.draw(G, node_size=70, node_color=cmap, with_labels=True)
    # plt.show()

    # nx.write_edgelist(G, 'test', data=False, delimiter=',')

    # plt.rc('font', family='serif')
    # plt.imshow(data["A"], interpolation="none")
    # plt.title("Adjacency Matrix")
    # # plt.colorbar()
    # plt.show()
    # plt.imshow(data["X"], aspect='auto', interpolation='none')
    # plt.title("Feature Matrix")
    # plt.colorbar()
    # plt.show()
    # nx.draw(G, node_size=140, with_labels=True)
    # plt.show()

    # cmap = []
    # for n in G.nodes:
    #     if any( G.nodes[n]["feature_vector"] == 1):
    #         cmap.append('red')
    #     else:
    #         cmap.append('blue')
    # nx.draw(G, node_size=70, node_color=cmap, with_labels=False)
    # plt.show()

    # sub_G = nx.MultiDiGraph()
    # def create_subgraph(G, sub_G, start_node):
    #     for n in G.successors(start_node):
    #         sub_G.add_edge(start_node, n)
    #         create_subgraph(G, sub_G, n)

    # create_subgraph(G, sub_G, 67)
    # attr_mapping = { n : data["X"][n] for n in sub_G.nodes() }
    # nx.set_node_attributes(sub_G, attr_mapping, "feature_vector")
    # print(sub_G.nodes())
    # cmap = []
    # for n in sub_G.nodes:
    #     if any( sub_G.nodes[n]["feature_vector"] == 1 ):
    #         cmap.append('red')
    #     else:
    #         cmap.append('blue')
    # nx.draw(sub_G, node_size=210, node_color=cmap, with_labels=False)
    # plt.show()