import argparse, random, os, sys, time, datetime, inspect
import numpy as np
import yaml
from matplotlib import pyplot as plt

import torch.nn
import torch.optim
import torch_geometric.data

from gnn import DGCNN


def main(train_dir, epochs):
    BATCHSIZE = 28
    DEVICE = torch.device("cuda:0")
    lr=0.8e-5#1e-5

    train_losses, valid_losses, valid_accuracies = [], [], []

    model = DGCNN(303, 2)
    model.to(DEVICE)

    criterion = torch.nn.NLLLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-06)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, eps=1e-06)

    source_model___init__ = inspect.getsource(model.__init__)
    source_model_forward = inspect.getsource(model.forward)
    source_lrdecay = inspect.getsource(adjust_learning_rate)
    params = ("lr: " + str(lr) + "\nLoss function: " + str(criterion) + "\nOptimiser: " + str(optimizer) + 
        "\nBatchsize: " + str(BATCHSIZE) +"\nlr decay: \n" + source_lrdecay)
    model_description = source_model___init__ + "\n" + source_model_forward + "\n\n" + params

    # Get data and divide into train and valid
    graph_locs = [ os.path.join(train_dir, filename) for filename in os.listdir(train_dir) if filename.endswith('.npz') ]
    random.shuffle(graph_locs)
    graph_locs_valid = graph_locs[:int(len(graph_locs)*0.2)]
    graph_locs_train = graph_locs[int(len(graph_locs)*0.2):]
    print("\nTraining set: {}".format(len(graph_locs_train)))
    print("Validation set: {}\n".format(len(graph_locs_valid)))

    # malware_cnt, benign_cnt = 0, 0
    # for lst in [graph_locs_train, graph_locs_valid]:
    #     for graph_loc in lst:
    #         graph_arrays = np.load(graph_loc)
    #         if graph_arrays["mal"]:
    #             malware_cnt += 1
    #         else:
    #             benign_cnt += 1
    # print("Malicious samples: {}".format(malware_cnt))
    # print("Benign samples: {}\n".format(benign_cnt))

    # Sort training data into batches
    batchsizes_train = [BATCHSIZE]*(int((len(graph_locs_train)/BATCHSIZE)))
    batchsizes_train.append(len(graph_locs_train) % BATCHSIZE)
    if batchsizes_train[-1] == 0: batchsizes_train.pop()

    # Train -- put this into a function --
    overtain_cntr = 0
    now = datetime.datetime.now().strftime("%d%m%Y-%H%M%S")
    start = time.time()
    for epoch in range(epochs):
        epoch_running_train_loss = 0.0

        model.train()

        # SHUFFLE THE TRAINING GRAPHS !!!!!!!!!!!!!!!!!!!
        random.shuffle(graph_locs_train)

        running_loss = 0.0
        graphs_for_batches = np.split(
            np.array(graph_locs_train), [ sum(batchsizes_train[:i]) for i in range(1, len(batchsizes_train)) ])
        for batch_idx, batch_files in enumerate(graphs_for_batches): # Training for one batch
            data_list = []

            for graph_loc in batch_files: # Load data for graphs in current batch
                graph_arrays = np.load(graph_loc)

                E = torch.LongTensor(graph_arrays['E']).to(DEVICE)
                E, _ = torch_geometric.utils.add_self_loops(E)
                X = torch.FloatTensor(graph_arrays['X'].astype(np.int16)).to(DEVICE)
                y = torch.LongTensor([1]).to(DEVICE) if graph_arrays['mal'] else torch.LongTensor([0]).to(DEVICE)

                data_list.append(torch_geometric.data.Data(x=X, edge_index=E, y=y))

            batch = torch_geometric.data.Batch().from_data_list(data_list)

            out = model(batch) 
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() # this loss is the average of the minibatch
            epoch_running_train_loss += loss.item()

            if (batch_idx + 1) % 40 == 0:
                print("[{}, {:2.2%}] loss: {:.8f}".format(epoch + 1, ((batch_idx + 1)*BATCHSIZE)/float(len(graph_locs_train)), running_loss/(40)))
                running_loss = 0.0
        
        train_losses.append((epoch_running_train_loss/len(graphs_for_batches)))

        adjust_learning_rate(optimizer, epoch, lr) # lr decay

        y_true, y_pred, y_pred_raw= [], [], []
        running_loss = 0.0
        model.eval()
        with torch.no_grad():
            for idx, graph_loc in enumerate(graph_locs_valid):
                graph_arrays = np.load(graph_loc)

                E = torch.LongTensor(graph_arrays['E']).to(DEVICE)
                E, _ = torch_geometric.utils.add_self_loops(E)
                X = torch.FloatTensor(graph_arrays['X'].astype(np.int16)).to(DEVICE)
                y = torch.LongTensor([1]).to(DEVICE) if graph_arrays['mal'] else torch.LongTensor([0]).to(DEVICE)

                b = torch.LongTensor(np.zeros(X.shape[0])).to(DEVICE) # Batch vector is all zeros since no batches for validation

                data = torch_geometric.data.Data(x=X, edge_index=E, y=y, batch=b)

                out = model(data)
                loss = criterion(out, data.y)

                running_loss += loss.item()

                y_pred_raw.append(out[0].detach().tolist())
                y_pred.append(out[0].max(0)[1].item())
                y_true.append(data.y.item())

        correct = len([ 1 for pred, true in zip(y_pred, y_true) if pred == true ])
        valid_losses.append(running_loss/len(graph_locs_valid))
        valid_accuracies.append(correct/len(graph_locs_valid))

        print("Validation loss: {:.4f}".format(running_loss/len(graph_locs_valid)))
        print("Binary accuracy on validation set: {}".format(correct/len(graph_locs_valid)))

        if epoch == 0:
            # torch.save(model.module.state_dict(), info["model_name"] + '_' + str(epoch + 1) + 'epochs.pth')
            old_valid_loss = valid_losses[0]
            valid_results = { 
                'model' : model_description,
                'best_epoch' : epoch + 1,
                'y_pred_best' : y_pred,
                'y_pred_best_raw' : y_pred_raw,
                'y_true' : y_true,
                'valid_graphs' : graph_locs_valid,
                'valid_losses' : valid_losses,
                'train_losses' : train_losses,
                }

        elif (valid_losses[-1] - old_valid_loss) < 0:
            # torch.save(model.module.state_dict(), info["model_name"] + '_' + str(epoch + 1) + 'epochs.pth')
            old_valid_loss = valid_losses[-1]
            overtain_cntr = 0 
            valid_results['best_epoch'] = epoch + 1
            valid_results['y_pred_best'] = y_pred
            valid_results['y_pred_best_raw'] = y_pred_raw
            valid_results['valid_losses'] = valid_losses
            valid_results['train_losses'] = train_losses

        else:
            overtain_cntr += 1
            valid_results['valid_losses'] = valid_losses
            valid_results['train_losses'] = train_losses

        with open('valid-results_{}.yaml'.format(now), 'w') as f:
            yaml.dump(valid_results, f)

        if overtain_cntr > 10:
            break

    end = time.time()

    print("###########################################")
    print("\nBest validation loss at epoch {}\n".format(epoch + 1 - overtain_cntr))
    print("Training losses:\n{}\n".format(train_losses))
    print("Validation losses:\n{}\n".format(valid_losses))
    print("Validation accuracies:\n{}\n".format(valid_accuracies))
    print("Best validation loss:{}\n".format(old_valid_loss))
    print("Average train time per epoch: {:.2f}s\n".format((end - start)/(epoch + 1 - overtain_cntr)))
    print("###########################################")


def adjust_learning_rate(optimizer, epoch, lr):
    lr = lr * (0.5 ** (epoch // 10)) 
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("train_dir")

    parser.add_argument("-e", "--epochs", nargs='?', type=int, default=10, action='store', dest='EPOCHS')

    args = parser.parse_args()

    return (args.train_dir, args.EPOCHS)

if __name__ == '__main__':
    arguments = parse_arguments()
    main(*arguments)
