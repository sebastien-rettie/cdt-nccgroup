# Extract characteristics from PE files into raw features
# Must place in benign/malign folder
# Assumes no duplicate files

# Can expand to include function imports

import pefile
import sys, os
import csv
import pandas as pd

# Folder to analyse
dir_path = os.path.dirname(os.path.realpath(__file__))

if "malware" in dir_path:
    malware = True
    print("viruses!")
elif "benign" in dir_path:
    malware = False
    print("all good")

# Grab paths of all executables in folder
file_list = []
dataframe_list = []

for folder, subfolder, files in os.walk(dir_path):
    for f in files:
        if ("extract-pe.py" not in f) and ("test.csv" not in f):
            # ignore self
            full_path = os.path.join(folder, f)
            file_list.append(full_path)
print("file list=", file_list)

# todo: split file list into batches to parallise


def extract_header(HEADER):
    # where HEADER has format pe.OPTIONAL_HEADER
    header_dict = HEADER.dump_dict()
    item_list = []
    var_list = []
    for item in header_dict:
        if "Structure" in str(item):
            continue
        item_list.append(item)
        var_list.append(header_dict[item]["Value"])
    header_dict = dict(zip(item_list, var_list))
    return header_dict


# Iterate over executables in folder
for executable in file_list:
    print(executable)
    pe = pefile.PE(executable, fast_load=True)

    # Parse section headers, num of sections, ordering and other attributes
    header_dict = {}

    if pe.OPTIONAL_HEADER:
        header_dict.update(extract_header(pe.OPTIONAL_HEADER))

    if pe.NT_HEADERS:
        header_dict.update(extract_header(pe.NT_HEADERS))

    if pe.FILE_HEADER:
        header_dict.update(extract_header(pe.FILE_HEADER))

    # todo: flags variable could cause trouble, add error catching

    if pe.DOS_HEADER:
        header_dict.update(extract_header(pe.DOS_HEADER))

    sect_no = 0

    item_list = []
    var_list = []

    for section in pe.sections:
        section = section.dump_dict()
        for item in section:
            if "Structure" in str(item):
                continue
            item_numbered = item + str(sect_no)
            item_list.append(item_numbered)
            var_list.append(section[item]["Value"])
        """
        If decoding does not happen automatically, use section.Name.decode('utf-8')
        """

        sect_no += 1

    item_list.append("SampleName")
    head, tail = os.path.split(executable)
    tail = str(tail)
    tail.replace(" ", "")
    var_list.append(tail)

    section_dict = dict(zip(item_list, var_list))
    header_dict.update(section_dict)

    # Write to list of dataframes
    df = pd.DataFrame.from_dict(header_dict, orient="index")
    dataframe_list.append(df.T)


# todo: add extraction of imports


final_df = pd.concat(dataframe_list)
final_df = final_df.set_index("SampleName")
final_df.fillna(0, inplace=True)

if malware == True:
    final_df["IsMalware"] = 1
    final_df.to_csv("processed_malware.csv")
elif malware == False:
    final_df["IsMalware"] = 0
    final_df.to_csv("processed_benign.csv")
