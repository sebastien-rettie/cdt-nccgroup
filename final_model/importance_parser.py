import numpy as np
import pandas as pd

def name_separator(string):
    """
    Parses feature names to a list from the concatenated strings in the CSV, with input
    string: Feature concatenations found in file, e.g. 'Magic/Subsystem/ImageBase'

    Returns features in string parsed into list, e.g. ['Magic', 'Subsystem', 'ImageBase']
    """

    feature_list = [] #Add separated features

    while "/" in string:

        index = string.find("/") #Find first instance of separator
        if index == 0:
            break #String ends in separator, so if its index is 0, string is finished

        else:
            feature_list.append(string[:index]) #Append feature
            string = string[index+1:] #String is now starting from next feature

    return feature_list

####################################################

n_features = 30 #Number of top features to parse

print("XGBoost top features:\n")

try:
    f = open('XGBoost_features.txt', 'w')

except FileNotFoundError:
    f = open('XGBoost_features.txt', 'x')

df = pd.read_csv('feature_results.csv')
dropped = list(df.iloc[5])

prev_appended = []

index = 0

for i in reversed(dropped[int(-(n_features/2)):]): #Only parse features as given above
        
    l = name_separator(i)

    if index == (n_features/2)-1:
        f.write(str(l)[1:-1]) #indexing removes list brackets, leaving comma delimited vals
        f.close()
        
    #Each new string includes previously parsed features, this removes them
    l = [j for j in l if (j not in prev_appended)]

    prev_appended += l

    print(">>> ", l)

    print("\n====================================\n")

    index += 1
