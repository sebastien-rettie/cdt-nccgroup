import argparse, random, os, sys, time, datetime, inspect
import numpy as np
import yaml
from matplotlib import pyplot as plt

import torch.nn
import torch.optim
import torch_geometric.data

from gnn import DGCNN


def main(train_dir, valid_dir, epochs):
    BATCHSIZE = 28 # 48
    DEVICE = torch.device("cuda:0")
    lr=1.5e-5#0.8e-5

    train_losses, valid_losses, valid_accuracies = [], [], []

    model = DGCNN(303, 2)
    model.to(DEVICE)

    criterion = torch.nn.NLLLoss()#weight=torch.FloatTensor([2.051, 1]).to(DEVICE))
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-06)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, eps=1e-06)

    source_model___init__ = inspect.getsource(model.__init__)
    source_model_forward = inspect.getsource(model.forward)
    source_lrdecay = inspect.getsource(adjust_learning_rate)
    params = ("lr: " + str(lr) + "\nLoss function: " + str(criterion) + "\nOptimiser: " + str(optimizer) + 
        "\nBatchsize: " + str(BATCHSIZE) +"\nlr decay: \n" + source_lrdecay)
    model_description = source_model___init__ + "\n" + source_model_forward + "\n\n" + params

    # Get data and divide into train and valid
    graph_locs_train = [ os.path.join(train_dir, filename) for filename in os.listdir(train_dir) if filename.endswith('.npz') ]
    graph_locs_valid = [ os.path.join(valid_dir, filename) for filename in os.listdir(valid_dir) if filename.endswith('.npz') ]
    print("\nTraining set: {}".format(len(graph_locs_train)))
    print("Validation set: {}\n".format(len(graph_locs_valid)))

    """
    malware_cnt, benign_cnt = 0, 0
    for graph_loc in graph_locs_train:
        graph_arrays = np.load(graph_loc)
        if graph_arrays["mal"]:
            malware_cnt += 1
        else:
            benign_cnt += 1
    print("Training set:\nMalicious samples: {}".format(malware_cnt))
    print("Benign samples: {}\n".format(benign_cnt))
    # cg_processed_dset2: malware 30648, benign 14904
    # cfg_processed_dset: malware 30788, benign 12897

    malware_cnt, benign_cnt = 0, 0
    for graph_loc in graph_locs_valid:
        graph_arrays = np.load(graph_loc)
        if graph_arrays["mal"]:
            malware_cnt += 1
        else:
            benign_cnt += 1
    print("Validation set:\nMalicious samples: {}".format(malware_cnt))
    print("Benign samples: {}\n".format(benign_cnt))
    # cg_processed_dset2: malware 7630, benign 3757
    # cfg_processed_dset: malware 7152, benign 4019
    """

    # Sort training data into batches
    batchsizes_train = [BATCHSIZE]*(int((len(graph_locs_train)/BATCHSIZE)))
    batchsizes_train.append(len(graph_locs_train) % BATCHSIZE)
    batchsizes_valid = [BATCHSIZE]*(int((len(graph_locs_valid)/BATCHSIZE)))
    batchsizes_valid.append(len(graph_locs_valid) % BATCHSIZE)
    if batchsizes_train[-1] == 0: batchsizes_train.pop()
    if batchsizes_valid[-1] == 0: batchsizes_valid.pop()

    # Train
    overtain_cntr = 0
    now = datetime.datetime.now().strftime("%d%m%Y-%H%M%S")
    print("now: {}".format(now))
    start = time.time()
    for epoch in range(epochs):
        epoch_running_train_loss = 0.0

        adjust_learning_rate(optimizer, epoch, lr) # lr decay

        # if epoch > 1:
        #     adjust_learning_rate(optimizer, lr, valid_losses[-1], valid_losses[-2])

        model.train()

        random.shuffle(graph_locs_train)

        running_loss = 0.0
        graphs_for_batches = np.split(
            np.array(graph_locs_train), [ sum(batchsizes_train[:i]) for i in range(1, len(batchsizes_train)) ])

        
        extra_graphs_train = random.sample(graph_locs_train, 5684)
        extra_batchsizes_train = [BATCHSIZE]*(int((len(extra_graphs_train)/BATCHSIZE)))
        extra_graphs_for_batches = np.split(
            np.array(extra_graphs_train), [ sum(extra_batchsizes_train[:i]) for i in range(1, len(extra_batchsizes_train)) ])
        graphs_for_batches = extra_graphs_for_batches + graphs_for_batches
        

        for batch_idx, batch_files in enumerate(graphs_for_batches): # Training for one batch
            data_list = []

            for graph_loc in batch_files: # Load data for graphs in current batch
                graph_arrays = np.load(graph_loc)

                # if graph_arrays['E'].shape[1] < 10: continue

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

            if (batch_idx + 1) % 100 == 0:
                print("[{}, {:2.2%}] loss: {:.8f}".format(epoch + 1, ((batch_idx + 1)*BATCHSIZE)/float(len(graph_locs_train)), running_loss/(100)))
                running_loss = 0.0
        
        train_losses.append((epoch_running_train_loss/len(graphs_for_batches)))

        y_true, y_pred, y_pred_raw = [], [], []
        running_loss = 0.0
        model.eval()
        graphs_for_batches = np.split(
            np.array(graph_locs_valid), [ sum(batchsizes_valid[:i]) for i in range(1, len(batchsizes_valid)) ])
        with torch.no_grad():
            for batch_idx, batch_files in enumerate(graphs_for_batches): # Training for one batch
                data_list = []

                for graph_loc in batch_files:
                    graph_arrays = np.load(graph_loc)

                    # if graph_arrays['E'].shape[1] < 10: continue

                    E = torch.LongTensor(graph_arrays['E']).to(DEVICE)
                    E, _ = torch_geometric.utils.add_self_loops(E)
                    X = torch.FloatTensor(graph_arrays['X'].astype(np.int16)).to(DEVICE)
                    y = torch.LongTensor([1]).to(DEVICE) if graph_arrays['mal'] else torch.LongTensor([0]).to(DEVICE)
                    
                    data_list.append(torch_geometric.data.Data(x=X, edge_index=E, y=y))
            
                batch = torch_geometric.data.Batch().from_data_list(data_list)

                out = model(batch)
                loss = criterion(out, batch.y)

                running_loss += loss.item()

                for pair in out.detach().tolist():
                    y_pred_raw.append(pair)
                    y_pred.append(pair.index(max(pair)))
                for y in batch.y.detach().tolist():
                    y_true.append(y)

        correct = len([ 1 for pred, true in zip(y_pred, y_true) if pred == true ])
        valid_losses.append(running_loss/len(graphs_for_batches))
        valid_accuracies.append(correct/len(graph_locs_valid))

        print("Validation loss: {:.4f}".format(running_loss/len(graphs_for_batches)))
        print("Binary accuracy on validation set: {}".format(correct/len(graph_locs_valid)))

        if epoch == 0:
            torch.save(model.state_dict(), "model_{}_{}.pth".format(now, valid_dir.split('/')[-3]))
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
            torch.save(model.state_dict(), "model_{}_{}.pth".format(now, valid_dir.split('/')[-3]))
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

        with open('model-info_{}_{}.yaml'.format(now, valid_dir.split('/')[-3]), 'w') as f:
            yaml.dump(valid_results, f)

        if overtain_cntr > 3: # > 10
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
    lr = lr * (0.8 ** (epoch // 2)) # // * ( 0.75, // 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# def adjust_learning_rate(optimizer, lr, valid_loss, old_valid_loss):
#     if valid_loss > (old_valid_loss*1.01): # decay lr if plateauing
#         lr = 0.8 * lr
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("train_dir")
    parser.add_argument("valid_dir")

    parser.add_argument("-e", "--epochs", nargs='?', type=int, default=10, action='store', dest='EPOCHS')

    args = parser.parse_args()

    return (args.train_dir, args.valid_dir, args.EPOCHS)

if __name__ == '__main__':
    arguments = parse_arguments()
    main(*arguments)
