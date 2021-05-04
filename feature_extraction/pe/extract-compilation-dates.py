import pefile
import os
import datetime
import pandas as pd
import matplotlib.pyplot as plt

# Extracts and plots compilation dates for all programs in folder

file_list = []
path_list = []
dates_list = []

dir_path = os.path.dirname(os.path.realpath(__file__))

for folder, subfolder, files in os.walk(dir_path):
    for f in files:
        if (
            ("extract-compilation-date.py" not in f)
            and ("benign-compilation-year.csv" not in f)
            and ("output.png" not in f)
        ):
            # ignore self
            full_path = os.path.join(folder, f)
            path_list.append(full_path)
            file_list.append(f)

for program in path_list:
    pe = pefile.PE(program, fast_load=True)

    file_header_dict = pe.FILE_HEADER.dump_dict()
    string_date = file_header_dict["TimeDateStamp"]["Value"]
    parsed_date = string_date[string_date.find("[") + 1 : string_date.find("]")]
    # format is Tue Jul 28 09:03:32 2020 UTC
    date_object = datetime.datetime.strptime(parsed_date, "%a %b %d %H:%M:%S %Y %Z")
    dates_list.append(date_object.year)

compile_dates = list(zip(file_list, dates_list))

dates = pd.DataFrame.from_records(compile_dates, columns=["name", "compilation-year"])

dates = pd.read_csv("benign-compilation-year.csv", usecols=[1, 2])

dates = dates.drop(dates[dates["compilation-year"] > 2021].index)

plot = dates.groupby(dates["compilation-year"]).count().plot(kind="bar", legend=None)

dates.to_csv("benign-compilation-year.csv")

fig = plot.get_figure()

fig.savefig("output.png")
