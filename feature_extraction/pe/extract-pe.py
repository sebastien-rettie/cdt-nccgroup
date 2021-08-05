# Extract characteristics from PE files into raw features
# Must place in benign/malign folder
# Assumes no duplicate files

# Can expand to include function imports

# now with entropy

from typing import final
import pefile
import sys, os
import csv
import pandas as pd
import datetime

# Folder to analyse
dir_path = os.path.dirname(os.path.realpath(__file__))

if "malware" in dir_path:
    malware = True
elif "benign" in dir_path:
    malware = False

# Grab paths of all executables in folder
file_list = []
dataframe_list = []

error_list = []

for folder, subfolder, files in os.walk(dir_path):
    for f in files:
        if (
            ("extract-pe.py" not in f)
            and ("processed_benign.csv" not in f)
            and ("error_list.txt" not in f)
        ):
            # ignore self
            full_path = os.path.join(folder, f)
            file_list.append(full_path)


def extract_header(HEADER):
    # where HEADER has format pe.OPTIONAL_HEADER
    header_dict = HEADER.dump_dict()
    item_list = []
    var_list = []
    for item in header_dict:
        if "Structure" in str(item):
            continue
        elif item == "TimeDateStamp":
            string_date = header_dict["TimeDateStamp"]["Value"]
            parsed_date = string_date[string_date.find("[") + 1 : string_date.find("]")]
            date_object = datetime.datetime.strptime(
                parsed_date, "%a %b %d %H:%M:%S %Y %Z"
            )
            item_list.append(item)
            var_list.append(date_object.year)
            break
        item_list.append(item)
        var_list.append(header_dict[item]["Value"])
    header_dict = dict(zip(item_list, var_list))
    return header_dict


# Iterate over executables in folder
for executable in file_list:
    print(executable)
    try:
        pe = pefile.PE(executable, fast_load=True)

        # Parse section headers, num of sections, ordering and other attributes
        header_dict = {}

        if pe.OPTIONAL_HEADER:
            header_dict.update(extract_header(pe.OPTIONAL_HEADER))

        if pe.NT_HEADERS:
            header_dict.update(extract_header(pe.NT_HEADERS))

        if pe.FILE_HEADER:
            header_dict.update(extract_header(pe.FILE_HEADER))

        if pe.DOS_HEADER:
            header_dict.update(extract_header(pe.DOS_HEADER))

        sect_no = 0

        item_list = []
        var_list = []

        for section in pe.sections:

            entropy = section.get_entropy()

            entropy_key = "Entropy" + str(sect_no)
            item_list.append(entropy_key)
            var_list.append(entropy)

            section = section.dump_dict()
            for item in section:
                if "Structure" in str(item):
                    continue
                elif str(item) == "Name":
                    section_name = section[item]["Value"]
                    section_name = section_name.replace("\\x00", "")
                    item_numbered = item + str(sect_no)
                    item_list.append(item_numbered)
                    var_list.append(section_name)

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
        tail = tail.replace(" ", "")
        var_list.append(tail)

        section_dict = dict(zip(item_list, var_list))
        header_dict.update(section_dict)

        # Write to list of dataframes
        df = pd.DataFrame.from_dict(header_dict, orient="index")
        dataframe_list.append(df.T)
    except pefile.PEFormatError as e:
        print("Parsing error!", e)
        head, tail = os.path.split(executable)
        tail = str(tail)
        error_filename = tail.replace(" ", "")
        print("Adding ", error_filename, "to error list.")
        error_list.append(error_filename)


final_df = pd.concat(dataframe_list)
parsed_files = final_df["SampleName"]
final_df = final_df.set_index("SampleName")

# XGBoost can handle nans
# final_df.fillna(0, inplace=True)


# This to stop /n characters causing trouble
final_df["e_res2"] = final_df["e_res2"].apply(lambda x: x.replace("\r\n", "\\r\\n"))
# might need to do it across dataframe

if error_list:
    with open("error_list.txt", "w") as f:
        for item in error_list:
            f.write("%s\n" % item)

malware = True

if malware == True:
    final_df["IsMalware"] = 1

    with open("processed_malware.csv", mode="w", newline="\n") as f:
        final_df.to_csv(f, sep=",", line_terminator=os.linesep, encoding="utf-8")

    with open("parsed_file_list.csv", mode="w", newline="\n") as f:
        parsed_files.to_csv(
            f, index=False, sep=",", line_terminator=os.linesep, encoding="utf-8"
        )


elif malware == False:
    final_df["IsMalware"] = 0

    with open("processed_benign.csv", mode="w", newline="\n") as f:
        final_df.to_csv(f, sep=",", line_terminator=os.linesep, encoding="utf-8")
