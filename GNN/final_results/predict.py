import sys, argparse, csv, os
import yaml
import numpy as np
import torch, torch_geometric

sys.path.append("/unix/cdtdisncc/2021/cfg/classifier")
from gnn import DGCNN

def main(weights, test_dir, output_name, pred_csv):
    DEVICE=('cpu')

    model = DGCNN(303, 2).to(DEVICE)
    pretrained_dict = torch.load(weights, map_location='cpu')
    pretrained_dict = {key.replace("module.", ""): value for key, value in pretrained_dict.items()}
    # ^^ In case DataParallel was used in training ^^
    model.load_state_dict(pretrained_dict)

    graph_locs_test = [ os.path.join(test_dir, filename) for filename in os.listdir(test_dir) if filename.endswith('.npz') ]

    y_pred, y_pred_raw, y_true = [], [], []
    model.eval()
    with torch.no_grad():
        for idx, graph_loc in enumerate(graph_locs_test):
            graph_arrays = np.load(graph_loc)

            E = torch.LongTensor(graph_arrays['E']).to(DEVICE)
            E, _ = torch_geometric.utils.add_self_loops(E)
            X = torch.FloatTensor(graph_arrays['X'].astype(np.int16)).to(DEVICE)
            # y = torch.LongTensor(graph_arrays['family']).to(DEVICE)
            y = torch.LongTensor([1]).to(DEVICE) if graph_arrays['mal'] else torch.LongTensor([0]).to(DEVICE)
            b = torch.LongTensor(np.zeros(X.shape[0])).to(DEVICE) # Batch vector is all zeros for batch of one
        
            data = torch_geometric.data.Data(x=X, edge_index=E, y=y, batch=b)

            out = model(data)

            y_pred_raw.append(out[0].detach().tolist())
            y_pred.append(out[0].max(0)[1].item())
            y_true.append(data.y.item())

            if ((idx + 1) % 500 == 0): print("{:2.2%}".format(float(idx + 1)/float(len(graph_locs_test))))

    results = {
        "test_graphs" : graph_locs_test,
        "y_true" : y_true,
        "y_pred" : y_pred,
        "y_pred_raw" : y_pred_raw
    }

    with open(output_name + ".yaml", 'w') as f:
        yaml.dump(results, f)

    if pred_csv:
        graph_names = [ os.path.basename(graph_loc)[:-4] for graph_loc in graph_locs_test ]
        y_pred_prob = [ np.exp(pair[1]) for pair in y_pred_raw ]

        with open(output_name + "_ensemble.csv", 'w') as f:
            writer = csv.writer(f)
            writer.writerows(zip(graph_names, y_pred_prob))


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("weights")
    parser.add_argument("test_dir")
    parser.add_argument("output_name")

    parser.add_argument("--pred_csv", action='store_true')

    args = parser.parse_args()

    return (args.weights, args.test_dir, args.output_name, args.pred_csv)
    

if __name__ == '__main__':
    arguments = parse_arguments()
    main(*arguments)