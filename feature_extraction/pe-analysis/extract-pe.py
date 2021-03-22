# Extract characteristics from PE files using 'pefile' into raw features

# Must place in benign/malign folder
# Written and tested on windows- convert to unix
# Assumes no duplicate files

# Can expand to include function imports
# Could be parallelised

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

# Could chop file list into batches/parallise here


def extract_header(HEADER):
    # where HEADER has format pe.OPTIONAL_HEADER
    header_dict = HEADER.dump_dict()
    item_list = []
    var_list = []
    for item in header_dict:
        if "Structure" in str(item):
            continue
        item_list.append(item)
        """
        try:
            value = header_dict[item]['Value'].decode('utf-8')
            var_list.append(value)
        except ValueError:
            var_list.append(header_dict[item]['Value'])
        except AttributeError:
            var_list.append(header_dict[item]['Value'])
        """
        var_list.append(header_dict[item]["Value"])
    header_dict = dict(zip(item_list, var_list))
    return header_dict


exe_no = 0

# Iterate over executables in folder
for executable in file_list:
    print(executable)
    pe = pefile.PE(executable, fast_load=True)
    # Parse all of NT_HEADER, FILE_HEADER, OPTIONAL_HEADER AND DOS_HEADER

    header_dict = {}

    if pe.OPTIONAL_HEADER:
        header_dict.update(extract_header(pe.OPTIONAL_HEADER))

    if pe.NT_HEADERS:
        header_dict.update(extract_header(pe.NT_HEADERS))

    if pe.FILE_HEADER:
        header_dict.update(extract_header(pe.FILE_HEADER))

    # Flags var could cause trouble here

    if pe.DOS_HEADER:
        header_dict.update(extract_header(pe.DOS_HEADER))

    # Parse section headers, structure, numb of sections, ordering,  name and other attributes

    # section_number = len(pe.sections)
    sect_no = 0

    item_list = []
    var_list = []

    for section in pe.sections:
        # print('Section number ',sect_no)
        section = section.dump_dict()
        for item in section:
            if "Structure" in str(item):
                continue
            item_numbered = item + str(sect_no)
            item_list.append(item_numbered)
            var_list.append(section[item]["Value"])
            # print(section[item]['Value'])
        # print(section)
        """
        # not sure if should try and deal with encodings or not
        print(section.Name), print(section.Name.decode('utf-8')),
        print (section.Name, hex(section.VirtualAddress),
        hex(section.Misc_VirtualSize), section.SizeOfRawData )
        """

        sect_no += 1

    section_dict = dict(zip(item_list, var_list))
    header_dict.update(section_dict)
    # print(header_dict)
    # Write to list of dataframes
    df = pd.DataFrame.from_dict(header_dict, orient="index")

    dataframe_list.append(df.T)

    exe_no += 1

# pe =  pefile.PE(r'C:\Users\Emily\Documents\ncc_group_project\malware-fake\OfficeSetup.exe', fast_load=True)

# Add import analysis here


final_df = pd.concat(dataframe_list)
final_df.fillna(0, inplace=True)

if malware == True:
    final_df["IsMalware"] = 1
elif malware == False:
    final_df["IsMalware"] = 0

final_df.to_csv("test.csv", index=False)

# print(pe.dump_info())
