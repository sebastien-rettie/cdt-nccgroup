import csv
import os
from tqdm import tqdm

n = 2   # length of n-gram
ngram_presence_file = f"{n}grams/{n}gram_presence_2021-04-28_23-34-13.csv"
test_list_file = "test_files_alex.csv"

outfile_train = f"{n}grams/{n}gram_presence_train.csv"
outfile_test = f"{n}grams/{n}gram_presence_test.csv"

def get_basenames(name_list):  # "base name" here means the name without the path and extension
    return [os.path.splitext(os.path.basename(name))[0] for name in name_list]

with open(test_list_file) as f:
    test_files = get_basenames(f.read().splitlines())

with open(ngram_presence_file) as f:
    fileout_train = open(outfile_train, "w")
    fileout_test = open(outfile_test, "w")
    writer_train = csv.writer(fileout_train, delimiter=",")
    writer_test = csv.writer(fileout_test, delimiter=",")

    csv_reader = csv.reader(f)
    header = next(csv_reader)
    writer_train.writerow(header)
    writer_test.writerow(header)

    for row in tqdm(csv_reader):
        if os.path.splitext(os.path.basename(row[0]))[0] in test_files:
            writer_test.writerow(row)
        else:
            writer_train.writerow(row)
    
    fileout_train.close()
    fileout_test.close()