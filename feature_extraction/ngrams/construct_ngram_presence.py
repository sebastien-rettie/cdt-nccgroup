"""
Construct a table indicating whether each n-gram listed in the distinct n-grams file is present in each binary file.

The output is saved in a CSV file with a row for each binary file and a column for each n-gram.
1 indicates that the n-gram is present in the file, and 0 that it is not.

To specify which binary files should be checked, specify either:
- a list of directories containing the n-grams for each file
  (set this as the value of the dirs variable)
  (each directory should have a separate file with n-grams for each binary file)
- the file containing a list of files to be checked
  (set this as the value of the list_file variable)
  (each file in the list should contain the n-grams for one binary file)
- both of these
If only one of these is specified, the other should be set to None, an empty list, or an empty string.
"""

import os
import csv
from tqdm import tqdm
from datetime import datetime

n = 2   # length of n-gram
distinct_ngrams_file = f"{n}grams/distinct_2grams_total.txt"
#list_file = "files_list.txt"
list_file = ""
dirs = [f"{n}grams/benign_distinct", f"{n}grams/malware_distinct"]
output_file = f"{n}grams/{n}gram_presence_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".csv"

with open(distinct_ngrams_file) as f:
    ngrams = set(f.read().splitlines())

files = []
if list_file:
    with open(list_file) as f:
        files += f.read().splitlines()

if dirs:
    for dir in dirs:
        files += [dir + "/" + file for file in sorted(os.walk(dir))[0][2]]

with open(output_file, "w") as outfile:
    writer = csv.writer(outfile, delimiter=",")
    writer.writerow(["file", "class"] + list(ngrams))
    for file in tqdm(files):
        with open(file) as f:
            file_ngrams = set(f.read().splitlines())
        
        base_name = os.path.splitext(os.path.basename(file))[0]  # remove path and .txt extension from binary file name
        class_label = 1 if base_name.split("_")[0]=="VirusShare" else 0
        writer.writerow([base_name, class_label] + [1 if ngram in file_ngrams else 0 for ngram in ngrams])
