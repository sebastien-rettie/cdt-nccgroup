"""
Create file which has unique ngrams from all files in input directory.
The input directory is the output directory of distinct_ngrams.py.
"""

import os
from tqdm import tqdm

input_dir = "hex_distinct"
output_file = "distinct_ngrams_total.txt"

filenames = sorted(os.walk(input_dir))[0][2]

distinct_n_grams = set()

for filename in tqdm(filenames):
    with open(input_dir + "/" + filename) as ngram_file:
        file_n_grams = ngram_file.readlines()
    distinct_n_grams.update(set(file_n_grams))

fileout = open(output_file,"w")
for n_gram in distinct_n_grams:
    fileout.write(n_gram)

fileout.close()
