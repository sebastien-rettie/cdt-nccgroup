import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time

datasize = 5672

def proportional_sample(year_range, data, n):
    """
    Identifies the dataset distribution and samples to size n while maintaining distribution

    - year_range: 2 item list (or tuple), start and end year + 1 (last not inclusive, like range()) for sampling
    - data: dataset to sample
    - n: size of dataset. 
    
    WARNING: n WILL NOT BE EXACT. input n will output between n and n+len(year_range), due to ceiling functions
    Returns 'data', downsampled dataset
    """

    if n > len(data):
        print('\nYou have provided a size n which is bigger than the dataset size you\'ve provided. Cannot downsample. Returning data...')
        return data

    print('\nDownsampling according to distribution...\n')

    original_datasize = 0 #Size of 'data' input
    year_sizes = [] #Length of each year's data

    for year in range(year_range[0], year_range[1], 1):
        frame = data.loc[data["TimeDateStamp"].str.contains(str(year))] #Get all rows from the data with this year
        original_datasize += len(frame)
        year_sizes.append(len(frame))

    datasizes = {} #Build dict with year/datasize key/pairs for range
    current_year = year_range[0] #For iterating through

    for size in year_sizes:
        datasizes[current_year] = np.ceil((size/original_datasize)*n) #Sample size according to year weight
        current_year += 1

    print('Distribution built:')
    print(datasizes)

    year_frames = []#Datasets to be concatenated

    for year in range(year_range[0], year_range[1], 1):
        frame = data.loc[data["TimeDateStamp"].str.contains(str(year))] #Get all rows from the data with this year
        frame = frame.sample(n=int(datasizes[year])) #And downsample
        year_frames.append(frame)

    print('\nData downsampled.\n')

    return pd.concat(year_frames)

def trimForDate(string):
    end = string.find('UTC')
    return string[end-5:end-1]

def generate_types(datafile):
    col_names = pd.read_csv(datafile, nrows=0).columns
    dtypes = {col: "str" for col in col_names}
    string_columns = [
        "SampleName",
        "Name0",
        "Name1",
        "Name10",
        "Name11",
        "Name12",
        "Name13",
        "Name14",
        "Name15",
        "Name16",
        "Name17",
        "Name18",
        "Name19",
        "Name2",
        "Name20",
        "Name21",
        "Name22",
        "Name23",
        "Name24",
        "Name25",
        "Name26",
        "Name27",
        "Name28",
        "Name29",
        "Name30",
        "Name3",
        "Name4",
        "Name5",
        "Name6",
        "Name7",
        "Name8",
        "Name9",
        "TimeDateStamp",
        "e_res",
        "e_res2",
    ]
    for column in string_columns:
        dtypes.update({column: "object"})
    return dtypes

train_file = 'train.csv'
start_time = time.time()
df_train = pd.read_csv(
    train_file,
    dtype=generate_types(train_file),
    engine="python",
)

counts = []
for year in range(1970, 2019, 1):
    counts.append(len(df_train.loc[df_train["TimeDateStamp"].str.contains(str(year))]))

plt.figure()
plt.plot(range(1970,2019,1), counts, label="before sample")

df_sampled = proportional_sample([2000,2019],df_train,983)
print(len(df_sampled))

counts_sample = []
for year in range(2000, 2019, 1):
    counts_sample.append(len(df_sampled.loc[df_sampled["TimeDateStamp"].str.contains(str(year))]))

plt.plot(range(2000,2019,1), counts_sample, label="after sample")
plt.legend(loc='best')
plt.show();

df_train.set_index(["SampleName"], inplace=True)

#     print(year, ':', len(df_train))
#     sum += len(df_train)
#     year_sizes.append(len(df_train))

# data = {}
# year = 2000

# for size in year_sizes:
#     data[year] = np.ceil((size/sum)*datasize)
#     year += 1

# plt.figure()
# plt.plot(range(2000,2019,1), year_sizes, label="before downsample");

# print('sampling...')

# year_frames = []

# for year in range(2000,2019,1):
#     train_file = 'csvs/{0}.csv'.format(year)
#     start_time = time.time()
#     df = pd.read_csv(
#         train_file,
#         dtype=generate_types(train_file),
#         engine="python",
#     )

#     df.set_index(["SampleName"], inplace=True)
#     df_sampled = df.sample(n=int(data[year]))

#     year_frames.append(df_sampled)

# df_train = pd.concat(year_frames)

# print(len(df_train))

# years = []
# for i in list(df_train["TimeDateStamp"]):
#         years.append(trimForDate(i))

# uniques = list(set(years))
# uniques.sort()
# sample_nos = []
# print(uniques)

# for u in uniques:
#     sample_nos.append(years.count(u))

# plt.plot(range(2000,2019,1), sample_nos, label="after downsample")
# plt.legend(loc='best');


