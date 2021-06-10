"""
For each file in input directory (which contains hexadecimal representations of executables),
find unique n-grams and save them in a new file of the same name in output directory
"""

import os
from tqdm import tqdm

input_dir = "hex"
output_dir = "hex_distinct"
n = 2                          # n-gram size

filenames = sorted(os.walk(input_dir))[0][2]

for filename in tqdm(filenames):
    with open(input_dir + "/" + filename) as hex_file:
        hex_string = hex_file.read().replace("\n", "")

    n_grams = []
    for i in range(int(len(hex_string)/2)-1):
        n_grams.append(hex_string[i*2 : (i+n)*2])

    # create set to get unique values
    #distinct_n_grams = set(n_grams)
    distinct_n_grams = n_grams

    fileout = open(output_dir + "/" + filename,"w")
    for n_gram in distinct_n_grams:
        fileout.write(n_gram + "\n")
    fileout.close()