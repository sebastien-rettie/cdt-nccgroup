import argparse, os, collections
import numpy as np
import yaml
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, fbeta_score, accuracy_score, roc_auc_score, roc_curve, precision_score, recall_score
from scikitplot.metrics import plot_roc

plt.rc('font', family='serif')


def main(input_file, confusion_threshold, F1_threshold, F2_threshold, accuracy_threshold,
    precision_threshold, recall_threshold, ROC_curve, ROC_curve_save, AUC_ROC, probs, callgraphs):

    with open(input_file, 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    y_true, y_pred, y_pred_raw, graphs = data['y_true'], data['y_pred'], data["y_pred_raw"], data['test_graphs']

    # Basic stats
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

    if precision_threshold:
        threshold = float(precision_threshold)
        y_precision = [ 0 if L > np.log(threshold) else 1 for L, R in y_pred_raw ]
        precision = precision_score(y_true, y_precision)
        print("Precision score for threshold {}: {:.4f}\n".format(threshold, precision))

    if recall_threshold:
        threshold = float(recall_threshold)
        y_recall = [ 0 if L > np.log(threshold) else 1 for L, R in y_pred_raw ]
        recall = recall_score(y_true, y_recall)
        print("Recall score for threshold {}: {:.4f}\n".format(threshold, recall))

    if ROC_curve:
        print("ROC curve:\n")
        fig, ax = plt.subplots()
        y_pred_prob = [ np.exp(pair[1]) for pair in y_pred_raw ]
        fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
        plt.plot(fpr, tpr, 'b')
        plt.legend(loc='lower right')
        plt.plot([0,1],[0,1],'r--')
        plt.xlim([-0.0,1.0])
        plt.ylim([-0.0,1.0])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        # plot_roc(y_true, y_pred_prob, ax=ax, plot_micro=False, plot_macro=False)
        plt.show()

    if ROC_curve_save:
        y_pred_prob = [ np.exp(pair[1]) for pair in y_pred_raw ]
        fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
        data = { 'tpr' : tpr.tolist(), 'fpr' : fpr.tolist() }
        with open("ROC_curve_data.yaml", 'w') as f:
            yaml.dump(data, f)

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
    
    if callgraphs:
        CATEGORIES = '/unix/cdtdisncc/2021/cfg/allAPI_categories_techheaders_reduced.yaml'
        GRAPH_PREFIX = '/unix/cdtdisncc/2021/cfg/classifier/data/cg_processed_dset2/test'

        with open(CATEGORIES, 'r') as f:
            opcode_categories = yaml.load(f, Loader=yaml.FullLoader)
        category_columns = { i : category for i, category in enumerate(opcode_categories.keys()) }
        # print(category_columns)

        benign_node_numbers_correct, benign_node_numbers_incorrect = [], []
        malware_node_numbers_correct, malware_node_numbers_incorrect = [], []
        benign_graph_density_correct, benign_graph_density_incorrect = [], []
        malware_graph_density_correct, malware_graph_density_incorrect = [], []
        benign_apis_correct, benign_apis_incorrect = [], []
        malware_apis_correct, malware_apis_incorrect = [], []
        for idx, graph in enumerate(graphs):
            data = np.load(os.path.join(GRAPH_PREFIX, os.path.basename(graph)))
            X = data['X']
            E = data['E']

            if y_true[idx] == 0:
                if y_true[idx] == y_pred[idx]:
                    benign_node_numbers_correct.append(X.shape[0])
                    benign_graph_density_correct.append(E.shape[1]/(X.shape[0]*(X.shape[0] - 1)))

                    for row in X:
                        for hit in np.where(row == 1)[0]:
                            benign_apis_correct.append(category_columns[hit])

                elif y_true[idx] != y_pred[idx]:
                    benign_node_numbers_incorrect.append(X.shape[0])
                    benign_graph_density_incorrect.append(E.shape[1]/(X.shape[0]*(X.shape[0] - 1)))

                    for row in X:
                        for hit in np.where(row == 1)[0]:
                            benign_apis_incorrect.append(category_columns[hit])

            elif y_true[idx] == 1:
                if y_true[idx] == y_pred[idx]:
                    malware_node_numbers_correct.append(X.shape[0])
                    malware_graph_density_correct.append(E.shape[1]/(X.shape[0]*(X.shape[0] - 1)))

                    for row in X:
                        for hit in np.where(row == 1)[0]:
                            malware_apis_correct.append(category_columns[hit])

                elif y_true[idx] != y_pred[idx]:
                    malware_node_numbers_incorrect.append(X.shape[0])
                    malware_graph_density_incorrect.append(E.shape[1]/(X.shape[0]*(X.shape[0] - 1)))

                    for row in X:
                        for hit in np.where(row == 1)[0]:
                            malware_apis_incorrect.append(category_columns[hit])

        benign_apis_incorrect_cnt = collections.Counter(benign_apis_incorrect)
        benign_apis_correct_cnt = collections.Counter(benign_apis_correct)
        malware_apis_incorrect_cnt = collections.Counter(malware_apis_incorrect)
        malware_apis_correct_cnt = collections.Counter(malware_apis_correct)
        print("Benign:")
        print("benign_apis_incorrect_cnt.most_common(20) = {}\n".format(benign_apis_incorrect_cnt.most_common(20)))
        print("benign_apis_correct_cnt.most_common(20) = {}\n".format(benign_apis_correct_cnt.most_common(20)))
        print("Malware:")
        print("malware_apis_incorrect_cnt.most_common(20) = {}\n".format(malware_apis_incorrect_cnt.most_common(20)))
        print("malware_apis_correct_cnt.most_common(20) = {}\n".format(malware_apis_correct_cnt.most_common(20)))

        rows_incorrect = { pair[0] for pair in benign_apis_incorrect_cnt.most_common(20) }
        rows_correct = { pair[0] for pair in benign_apis_correct_cnt.most_common(20) }
        rows = list(rows_incorrect.union(rows_correct))
        rows.sort(key=lambda x : benign_apis_correct_cnt[x])
        rows.reverse()
        correct_total, incorrect_total = sum(benign_apis_correct_cnt.values()), sum(benign_apis_incorrect_cnt.values())
        cell_text = [ [round((benign_apis_correct_cnt[api]/correct_total)*100, 2), round((benign_apis_incorrect_cnt[api]/incorrect_total)*100, 2)] for api in rows ]
        cols = ["Correct classification", "Incorrect classification"]
        fig, ax = plt.subplots()
        fig.patch.set_visible(False)
        ax.axis('off')
        ax.axis('tight')
        ax.table(cellText=cell_text, rowLabels=rows, colLabels=cols, loc='center', edges='closed', fontsize=8)
        fig.tight_layout()
        plt.show()

        rows_incorrect = { pair[0] for pair in malware_apis_incorrect_cnt.most_common(20) }
        rows_correct = { pair[0] for pair in malware_apis_correct_cnt.most_common(20) }
        rows = list(rows_incorrect.union(rows_correct))
        rows.sort(key=lambda x : malware_apis_correct_cnt[x])
        rows.reverse()
        correct_total, incorrect_total = sum(malware_apis_correct_cnt.values()), sum(malware_apis_incorrect_cnt.values())
        cell_text = [ [round((malware_apis_correct_cnt[api]/correct_total)*100, 2), round((malware_apis_incorrect_cnt[api]/incorrect_total)*100, 2)] for api in rows ]
        cols = ["Correct classification", "Incorrect classification"]
        fig, ax = plt.subplots()
        fig.patch.set_visible(False)
        ax.axis('off')
        ax.axis('tight')
        ax.table(cellText=cell_text, rowLabels=rows, colLabels=cols, loc='center', edges='closed', fontsize=8)
        fig.tight_layout()
        plt.show()

        bins=np.histogram(np.hstack((np.array(benign_graph_density_correct),np.array(benign_graph_density_incorrect))), bins=40)[1]
        plt.hist(benign_graph_density_correct, bins=bins, histtype='step', label="Correct classification", density=True)
        plt.hist(benign_graph_density_incorrect, bins=bins, histtype='step', label="Incorrect classification", density=True)
        plt.legend(fontsize=18)
        plt.title("Graph density for correctly and incorrectly classified benign callgraphs", fontsize=20)
        plt.xlabel("Graph density of callgraph", fontsize=18)
        plt.ylabel("Density", fontsize=18)
        plt.show()

        bins=np.histogram(np.hstack((np.array(malware_graph_density_correct),np.array(malware_graph_density_incorrect))), bins=40)[1]
        plt.hist(malware_graph_density_correct, bins=bins, histtype='step', label="Correct classification", density=True)
        plt.hist(malware_graph_density_incorrect, bins=bins, histtype='step', label="Incorrect classification", density=True)
        plt.legend(fontsize=18)
        plt.title("Graph density for correctly and incorrectly classified malicious callgraphs", fontsize=20)
        plt.xlabel("Graph density of callgraph", fontsize=18)
        plt.ylabel("Density", fontsize=18)
        plt.show()

        bins=np.histogram(np.hstack((np.array(benign_node_numbers_correct),np.array(benign_node_numbers_incorrect))), bins=40)[1]
        plt.hist(benign_node_numbers_correct, bins=bins, histtype='step', label="Correct classification", density=True)
        plt.hist(benign_node_numbers_incorrect, bins=bins, histtype='step', label="Incorrect classification", density=True)
        plt.legend(fontsize=18)
        plt.title("Number of nodes for correctly and incorrectly classified benign callgraphs", fontsize=20)
        plt.xlabel("Number of nodes in callgraph", fontsize=18)
        plt.ylabel("Density", fontsize=18)
        plt.show()

        bins=np.histogram(np.hstack((np.array(malware_node_numbers_correct),np.array(malware_node_numbers_incorrect))), bins=40)[1]
        plt.hist(malware_node_numbers_correct, bins=bins, histtype='step', label="Correct classification", density=True)
        plt.hist(malware_node_numbers_incorrect, bins=bins, histtype='step', label="Incorrect classification", density=True)
        plt.legend(fontsize=18)
        plt.title("Number of nodes for correctly and incorrectly classified malicious callgraphs", fontsize=20)
        plt.xlabel("Number of nodes in callgraph", fontsize=18)
        plt.ylabel("Density", fontsize=18)
        plt.show()


def parse_arguments():

    parser = argparse.ArgumentParser(description="Analyse network performance.")
    parser.add_argument("input_file")

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
    parser.add_argument("--precision", nargs=None, type=str, default='',action='store',
        dest='PRECISION_THRESHOLD', help="Show precision for a given threshold")
    parser.add_argument("--recall", nargs=None, type=str, default='',action='store',
        dest='RECALL_THRESHOLD', help="Show recall for a given threshold")
    parser.add_argument("--ROC_curve", action='store_true', help="Plot ROC curve")
    parser.add_argument("--ROC_curve_save", action='store_true',
        help="Save the TPR and FPR required to plot the ROC_curve")
    parser.add_argument("--AUC_ROC", action='store_true', help="Show ROC AUC score")
    parser.add_argument("--probs", action='store_true', help="Plot histogram of probabilies")
    parser.add_argument("--callgraphs", action='store_true', 
        help="Examine some properties of the correctl/incorrect prediction validation set callgraphs")

    args = parser.parse_args()
    
    return (args.input_file, args.CONFUSION_THRESHOLD, args.F1_THRESHOLD, args.F2_THRESHOLD,
        args.ACCURACY_THRESHOLD, args.PRECISION_THRESHOLD, args.RECALL_THRESHOLD, args.ROC_curve,
        args.ROC_curve_save, args.AUC_ROC, args.probs, args.callgraphs)


if __name__ == "__main__":
    arguments = parse_arguments()
    main(*arguments)
