import pefile
import os
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from plotting import (
    plot_learning_curve,
    plot_validation_curve,
)
from preprocessing import (
    generate_types,
    get_ct_feature_names,
    preprocess_dataframe,
    encode_scale,
)

# Reading in preprocessed TRAIN data
train_file = "train.csv"
df_train = pd.read_csv(
    train_file,
    dtype=generate_types(train_file),
    engine="python",
)

df_train.set_index(["SampleName"], inplace=True)

"""
# Reading in preprocessed TEST data
test_file = "test.csv"
df_test = pd.read_csv(
    test_file,
    dtype=generate_types(test_file),
    engine="python",
)

df_test.set_index(["SampleName"], inplace=True)
"""

def trimForDate(string):

    for i in string:
        start = i.find('[')
        end = i.find(']')
        datestring = i[start+1:end]
    
    return datestring

#test_frames = [] #for test datasets

for i in np.arange(1970,2022,1):

    print('Processing year {0}...'.format(i))

    #find rows that contain timedate stamps including current year and save df to csv
    df_year = df_train.loc[df_train['TimeDateStamp'].str.contains('{0} UTC'.format(i))]
    df_year.to_csv('{0}.csv'.format(i))

    #test_frames.append(df_year)

    df_year = 0 #reset dataframe

#Concatenate test years and build csv
#df = pd.concat(test_frames)
#df.to_csv('2019-21_test.csv')
