# Graph Classifier
Here is the GNN model and scripts to prepare data for training, do training, and analyse results. All the data that I processed and used for the training can be found in compressed form at /unix/cdtdisncc/2021/cfg/classifier/data (its also on the s3 bucket). 

- `gnn.py`: this is the PyTorch DGCNN model.
- `train.py`: this is the training script (on the cdt GPU machines, email hep computing people to get access to gpu01 to use these) for the binary classification. There is a fair bit going on in this script
	- learning rate decay.
	- the model and training information is being saved to a yaml file (for the purpose of remembering whats going on when hyperparameter tuning).
	- Adding self-loops to the graphs, the GCN layers already do this but I found adding an extra self-loop improves the CG classifier's performance
	- Early stopping, saves model at best epoch
	- Random commented out stuff I was trying out.
- `train_multiclass.py`: same story as `train.py`but for malware family classification.
- validation_results/: this is where I was analysing the performance on the validation set for hyperparameter tuning and for plots to show the group. the `analyse.py`analyses the training information from `train.py`.
- final_results/: this is where I analysed the performance of the test set. Included are my saved final models. The `predict.py`gets the predictions of the model on the test set. The analyse scripts then produce plots and stuff. 
- process_graphs: scripts for processing the graph data. `process_cfg.py` counts all the opcodes to find the most frequent and then resizes the feature matrices to use the most frequent opcodes. It also converts the adjacency matrix into an edge list (I didnt realise the GCN layers took edge lists). The `process_cg.py`only converts adjacency matrix into edge list. `process_processed.py`just moves graphs with <= 5 nodes to a different folder. `get_family.py` uses the metadata files to give graphs malware familiy labels.

## Workflow:
This is for binary classification but basically the same idea for multiclass (I wish GitHub rendered Mermaid graphs, these would look :ok_hand:).
```
graphs --> process_cfg/cg.py --> process_processed.py --> train.py --> saved model --> predict.py --> anaylse_test.py
                                                           /  ^
                                                          /    \ 
                                   validation_results/ <--      hyperparameter tuning
                                           |                            ^
                                           |                            |
                                            ----------------------------
```
