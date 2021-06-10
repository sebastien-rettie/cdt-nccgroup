import os
import csv
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime

n = 2   # length of n-gram
# only use training sample to calculate information gain!
ngram_presence_file = f"{n}grams/{n}gram_presence_train.csv"
info_gain_file = f"{n}grams/{n}gram_info_gain_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".csv"

print("Loading dataset...")
data = pd.read_csv(ngram_presence_file, usecols=[1] + [i for i in range(45001,50001)], dtype=int)
print("Loaded")

n_samples = len(data)
p_benign = len(data[data["class"]==0])/n_samples
p_malicious = len(data[data["class"]==1])/n_samples
p_c = [p_benign, p_malicious]
print("Number of samples:", n_samples)
print("Fraction of samples benign samples:", p_benign)
print("Fraction of samples malware samples:", p_malicious)

IGs = []
non_zero_p_v_cs = []                  # this is to test why I am getting zero division error

for column in tqdm(data.columns[1:]):
    data_sub = data[["class", column]]
    IG = 0
    for v in [0,1]:
        p_v = len(data_sub[data_sub[column] == v])/n_samples
        for c in [0, 1]:
            data_c = data_sub[data_sub["class"] == c]
            data_v = data_c[data_c[column] == v]
            p_v_c = len(data_v)/len(data_c)

            if (p_v == 0 or p_c[c] == 0) and p_v_c != 0:    # this is to test why I am getting zero division error
                # I think this shouldn't be possible, that p_v is 0 but p_v_c is not
                print(column)
                print(p_v_c)
                non_zero_p_v_cs.append(p_v_c)
                pass
            else:
                el = 0 if p_v_c == 0 else p_v_c * np.log(p_v_c / p_v / p_c[c])
                IG += el
            
    IGs.append(IG)

with open(info_gain_file, "w") as f:
    csv_writer = csv.writer(f, delimiter=",")
    csv_writer.writerow(data.columns[1:])
    csv_writer.writerow(IGs)

with open(f"{n}grams/pvc_list.txt", "w") as f:                   # this is to test why I am getting zero division error
    for el in non_zero_p_v_cs:
        f.write(str(el) + "\n")