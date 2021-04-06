import os
import pandas as pd
import numpy as np
from tqdm import tqdm

unique_values_file = "distinct_ngrams_total.txt"
ngrams = pd.read_table(unique_values_file, names = ["ngram"])[:100] # ONLY 100 COLUMNS FOR TESTING PURPOSES, INCLUDE ALL LATER

dir_benign = "distinct_ngrams"
dir_malicious = "distinct_ngrams_malicious"
filenames_benign = sorted(os.walk(dir_benign))[0][2]
filenames_malicious = sorted(os.walk(dir_malicious))[0][2]
data_benign = pd.DataFrame(filenames_benign, columns=["file"])
data_malicious = pd.DataFrame(filenames_malicious, columns=["file"])

columns = ["file", "class"] + list(ngrams["ngram"])
data=pd.DataFrame(columns = columns)

n_best = 50   # number of best features to keep

output_file = "features_" + str(len(ngrams)) + ".csv"
output_file_best = "features_best_" + str(n_best) + ".csv"

for file in tqdm(filenames_benign):
    with open(dir_benign + "/" + file) as f:
        file_ngrams = f.read().splitlines()
    new_row = pd.Series([file, 0] + [1 if ngram in file_ngrams else 0 for ngram in ngrams["ngram"]], index=columns)
    data = data.append(new_row, ignore_index=True)

for file in tqdm(filenames_malicious):
    with open(dir_malicious + "/" + file) as f:
        file_ngrams = f.read().splitlines()
    new_row = pd.Series([file, 1] + [1 if ngram in file_ngrams else 0 for ngram in ngrams["ngram"]], index=columns)
    data = data.append(new_row, ignore_index=True)

n_files = len(filenames_benign) + len(filenames_malicious)
p_benign = len(filenames_benign)/n_files
p_malicious = len(filenames_malicious)/n_files
p_c = [p_benign, p_malicious]

IGs = []

for column in tqdm(data.columns[2:]):
    IG = 0
    for v in [0,1]:
        p_v = len(data[data[column] == v])/n_files
        for c in [0, 1]:
            data_c = data[data["class"] == c]
            data_v = data_c[data_c[column] == v]
            p_v_c = len(data_v)/len(data_c)

            el = 0 if p_v_c == 0 else p_v_c * np.log(p_v_c / p_v / p_c[c])
            IG += el
            
    IGs.append(IG)

data = data.append( pd.Series(["information gain", None] + IGs, index=columns), ignore_index=True)
data.to_csv(output_file)

IG_sorted = sorted(IGs)
threshold = IG_sorted[-n_best]

for column in tqdm(data.columns[2:]):
    if data[column].iloc[-1] < threshold:   # the last row is the information gain
        data.pop(column)
data.to_csv(output_file_best)