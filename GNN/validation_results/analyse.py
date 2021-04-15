import argparse, os, collections
import numpy as np
import yaml
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, fbeta_score, accuracy_score, roc_auc_score
from scikitplot.metrics import plot_roc

plt.rc('font', family='serif')


def main(input_file, print_model, confusion_threshold, F1_threshold, F2_threshold, accuracy_threshold,
    ROC_curve, AUC_ROC, probs, learning_curve, callgraphs):

    with open(input_file, 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    model, epoch, valid_losses, train_losses = data['model'], data['best_epoch'], data['valid_losses'], data['train_losses']
    y_true, y_pred, y_pred_raw, graphs = data['y_true'], data['y_pred_best'], data["y_pred_best_raw"], data['valid_graphs']
    # Some of the older valid-results use a different format
    # model, epoch, valid_losses, train_losses = data['model'], data['epoch'], data['valid_losses'], data['train_losses']
    # y_true, y_pred, y_pred_raw,= data['y_true'], data['y_pred'], data["y_pred_raw"]

    if print_model:
        print(model + '\n')

    # Basic stats
    print("Best epoch: {}".format(epoch))
    print("Best validation loss: {:.4f}".format(valid_losses[-1]))
    print("Best binary Accuracy for 0.5 threshold: {:.2f}%\n".format(
        (len([ 1 for pred, true in zip(y_pred, y_true) if pred == true ])*100)/len(y_true)))

    if confusion_threshold:
        threshold = float(confusion_threshold)
        print("Confusion matrix for threshold {}:\n".format(threshold))
        y_confusion = [ 0 if L > np.log(threshold) else 1 for L, R in y_pred_raw ]
        C = confusion_matrix(y_true, y_confusion)
        df_C = pd.DataFrame(C, range(2), range(2))

        group_counts = [ "{0:0.0f}".format(value) for value in C.flatten() ]
        group_percentages = [ "{0:.2%}".format(value) for value in C.flatten()/np.sum(C) ]
        labels = [ f"{v1}\n{v2}" for v1, v2 in zip(group_counts,group_percentages) ]
        labels = np.asarray(labels).reshape(2,2)

        sns.set(font_scale=1.4)
        ax = plt.subplot()
        sns.heatmap(C, annot=labels, annot_kws={'size': 16}, fmt='', ax=ax, cmap='Blues')
        ax.set_xlabel("Predicted labels")
        ax.set_ylabel("True labels")
        ax.set_title("Confusion Matrix") 
        ax.xaxis.set_ticklabels(["benign", "malware"])
        ax.yaxis.set_ticklabels(["benign", "malware"])
        plt.show()

    if F1_threshold == 'all':
        print("F1 scores:\n")
        F1s = []
        thresholds = np.linspace(0.01, 0.99, 99)
        for threshold in thresholds:
            y_F1 = [ 0 if L > np.log(threshold) else 1 for L, R in y_pred_raw ]
            F1s.append(fbeta_score(y_true, y_F1, beta=1))

        best_idx = F1s.index(max(F1s))

        plt.plot(thresholds, F1s)
        plt.vlines(thresholds[best_idx], ymin=0, ymax=1, color='r')
        plt.text(thresholds[best_idx] + 0.01, 0.5, "{:.4f}\n{}".format(F1s[best_idx], thresholds[best_idx]),
            color='r', fontsize=18)
        plt.xlabel("Threshold", fontsize=18)
        plt.ylabel("F1 Score", fontsize=18)
        plt.title("F1 score by threshold", fontsize=20)
        plt.show()
    
    elif F1_threshold:
        threshold = float(F1_threshold)
        y_F1 = [ 0 if L > np.log(threshold) else 1 for L, R in y_pred_raw ]
        F1 = fbeta_score(y_true, y_F1, beta=1)
        print("F1 score for threshold {}: {:.4f}\n".format(threshold, F1))

    if F2_threshold =='all':
        print("F2 scores:\n")
        F2s = []
        thresholds = np.linspace(0.01, 0.99, 99)
        for threshold in thresholds:
            y_F2 = [ 0 if L > np.log(threshold) else 1 for L, R in y_pred_raw ]
            F2s.append(fbeta_score(y_true, y_F2, beta=2))

        best_idx = F2s.index(max(F2s))

        plt.plot(thresholds, F2s)
        plt.vlines(thresholds[best_idx], ymin=0, ymax=1, color='r')
        plt.text(thresholds[best_idx] + 0.01, 0.5, "{:.4f}\n{}".format(F2s[best_idx], thresholds[best_idx]),
            color='r', fontsize=18)
        plt.xlabel("Threshold", fontsize=18)
        plt.ylabel("F2 Score", fontsize=18)
        plt.title("F2 score by threshold", fontsize=20)
        plt.show()
    
    elif F2_threshold:
        threshold = float(F2_threshold)
        y_F2 = [ 0 if L > np.log(threshold) else 1 for L, R in y_pred_raw ]
        F2 = fbeta_score(y_true, y_F2, beta=2)
        print("F2 score for threshold {}: {:.4f}\n".format(threshold, F2))

    if accuracy_threshold == 'all':
        print("Accuracies:\n")
        accs = []
        thresholds = np.linspace(0.01, 0.99, 99)
        for threshold in thresholds:
            y_acc = [ 0 if L > np.log(threshold) else 1 for L, R in y_pred_raw ]
            accs.append(accuracy_score(y_true, y_acc))

        best_idx = accs.index(max(accs))

        # L_idx = min(range(0, int(len(accs)/2)), key=lambda i: abs(accs[i]-0.95))
        # R_idx = min(range(int(len(accs)/2), len(accs)), key=lambda i: abs(accs[i]-0.95))

        plt.plot(thresholds, accs)
        plt.vlines(thresholds[best_idx], ymin=0, ymax=1, color='r')
        plt.text(thresholds[best_idx] + 0.01, 0.5, "{:.4f}\n{:.2f}".format(accs[best_idx], thresholds[best_idx]),
            color='r', fontsize=18)
        # plt.vlines(thresholds[L_idx], ymin=0, ymax=1, color='b')
        # plt.text(thresholds[L_idx] + 0.01, 0.5, "{:.4f}\n{:.2f}".format(accs[L_idx], thresholds[L_idx]),
        #     color='b', fontsize=18)
        # plt.vlines(thresholds[R_idx], ymin=0, ymax=1, color='b')
        # plt.text(thresholds[R_idx] + 0.01, 0.5, "{:.4f}\n{:.2f}".format(accs[R_idx], thresholds[R_idx]),
        #     color='b', fontsize=18)
        plt.xlabel("Threshold", fontsize=18)
        plt.ylabel("Accuracy", fontsize=18)
        plt.title("Accuracy score by threshold", fontsize=20)
        plt.show()

    elif accuracy_threshold:
        threshold = float(accuracy_threshold)
        y_acc = [ 0 if L > np.log(threshold) else 1 for L, R in y_pred_raw ]
        acc = accuracy_score(y_true, y_acc)
        print("Accuracy score for threshold {}: {:.4f}\n".format(threshold, acc))

    if ROC_curve:
        print("ROC curve:\n")
        fig, ax = plt.subplots()
        plot_roc(y_true, y_pred_raw, ax=ax, plot_micro=False, plot_macro=False)
        plt.show()

    if AUC_ROC:
        y_pred_prob = [ np.exp(pair[1]) for pair in y_pred_raw ] # roc_auc_score wants the probabilities of class 1 (malware)
        print("ROC AUC score: {:.4f}\n".format(roc_auc_score(np.array(y_true), np.array(y_pred_prob)))) 

    if probs:
        # y_pred_prob_malware = [ np.exp(probs[1]) for probs, true in zip(y_pred_raw, y_true) if true == 1 ]
        # y_pred_prob_benign = [ np.exp(probs[0]) for probs, true in zip(y_pred_raw, y_true) if true == 0 ]
        y_pred_prob_correct = [ np.exp(probs[1]) for probs, pred, true in zip(y_pred_raw, y_pred, y_true) if pred == true ]
        y_pred_prob_incorrect = [ np.exp(probs[1]) for probs, pred, true in zip(y_pred_raw, y_pred, y_true) if pred != true ]
        # y_pred_prob_FP = [ np.exp(probs[1]) for probs, pred, true in zip(y_pred_raw, y_pred, y_true) if (pred != true and pred == 1) ]
        # y_pred_prob_FN = [ np.exp(probs[1]) for probs, pred, true in zip(y_pred_raw, y_pred, y_true) if (pred != true and pred == 0) ]

        plt.hist(y_pred_prob_correct, bins=40, histtype='step', density=True, label="Correct prediction")
        plt.hist(y_pred_prob_incorrect, bins=40, histtype='step', density=True, label="Incorrect prediction")
        # plt.hist(y_pred_prob_FP, bins=40, histtype='step', density=True, label="FP")
        # plt.hist(y_pred_prob_FN, bins=40, histtype='step', density=True, label="FN")
        plt.xlabel("Probability of belonging to malware class", fontsize=18)
        plt.ylabel("Density", fontsize=18)
        plt.legend(fontsize=18)
        plt.show()
    
    if learning_curve:
        plt.plot([ e + 1 for e in list(range(len(train_losses)))], train_losses, label="Training")
        plt.plot([ e + 1 for e in list(range(len(valid_losses)))], valid_losses, label="Validation")
        plt.vlines(epoch, ymin=min(valid_losses) - 0.03, ymax=max(train_losses), color='r')
        plt.text(epoch + 0.01, max(valid_losses), "Epoch {}".format(epoch), color='r', fontsize=18)
        plt.legend(fontsize=18)
        plt.xlabel("Epoch", fontsize=18)
        plt.ylabel("Negative log likelihood", fontsize=18)
        plt.title("Learning Curve", fontsize=20)
        plt.show()

    if callgraphs:
        CATEGORIES = '/unix/cdtdisncc/2021/cfg/allAPI_categories_techheaders_reduced.yaml'
        GRAPH_PREFIX = '/unix/cdtdisncc/2021/cfg/classifier'

        with open(CATEGORIES, 'r') as f:
            opcode_categories = yaml.load(f, Loader=yaml.FullLoader)
        category_columns = { i : category for i, category in enumerate(opcode_categories.keys()) }
        # print(category_columns)

        node_numbers_correct, node_numbers_incorrect = [], []
        apis_correct, apis_incorrect = [], []
        for idx, graph in enumerate(graphs):
            data = np.load(os.path.join(GRAPH_PREFIX, graph))
            X = data['X']

            if y_true[idx] == y_pred[idx]:
                node_numbers_correct.append(X.shape[0])

                for row in X:
                    for hit in np.where(row == 1)[0]:
                        apis_correct.append(category_columns[hit])

            elif y_true[idx] != y_pred[idx]:
                node_numbers_incorrect.append(X.shape[0])

                for row in X:
                    for hit in np.where(row == 1)[0]:
                        apis_incorrect.append(category_columns[hit])

        apis_incorrect_cnt = collections.Counter(apis_incorrect)
        apis_correct_cnt = collections.Counter(apis_correct)
        print("apis_incorrect_cnt.most_common(20) = {}\n".format(apis_incorrect_cnt.most_common(20)))
        print("apis_correct_cnt.most_common(20) = {}\n".format(apis_correct_cnt.most_common(20)))

        plt.hist(node_numbers_correct, bins=40, histtype='step', label="Correct classification", density=True)
        plt.hist(node_numbers_incorrect, bins=40, histtype='step', label="Incorrect classification", density=True)
        plt.legend(fontsize=18)
        plt.title("Number of nodes for correctly and incorrectly classified callgraphs", fontsize=20)
        plt.xlabel("Number of nodes in callgraph", fontsize=18)
        plt.ylabel("Density", fontsize=18)
        plt.show()


def parse_arguments():

    parser = argparse.ArgumentParser(description="Analyse network performance.")
    parser.add_argument("input_file")

    parser.add_argument("-p", action='store_true', help="Print model")
    parser.add_argument("--confusion", nargs=None, type=str, default='', action='store',
        dest='CONFUSION_THRESHOLD', help="Show confusion matrix for given threshold")
    parser.add_argument("--F1", nargs=None, type=str, default='', action='store',
        dest='F1_THRESHOLD',
        help="Show F1 score for a given threshold or for range of thresholds if 'all'")
    parser.add_argument("--F2", nargs=None, type=str, default='', action='store',
        dest='F2_THRESHOLD',
        help="Show F2 score for a given threshold or for range of thresholds if 'all'")
    parser.add_argument("--accuracy", nargs=None, type=str, default='', action='store',
        dest='ACCURACY_THRESHOLD',
        help="Show accuracy for a given threshold or for range of thresholds if 'all'")
    parser.add_argument("--ROC_curve", action='store_true', help="Plot ROC curve")
    parser.add_argument("--AUC_ROC", action='store_true', help="Show ROC AUC score")
    parser.add_argument("--probs", action='store_true', help="Plot histogram of probabilies")
    parser.add_argument("--lc", action='store_true', help="Plot learning curve")
    parser.add_argument("--callgraphs", action='store_true', 
        help="Examine some properties of the correctly/incorrectly prediction validation set callgraphs")

    args = parser.parse_args()
    
    return (args.input_file, args.p, args.CONFUSION_THRESHOLD, args.F1_THRESHOLD, args.F2_THRESHOLD,
        args.ACCURACY_THRESHOLD, args.ROC_curve, args.AUC_ROC, args.probs, args.lc, args.callgraphs)


if __name__ == "__main__":
    arguments = parse_arguments()
    main(*arguments)
