import numpy as np
import pandas as pd

names = [
        'AdaBoost',
        'RandomForest',
        'DecisionTree',
        'XGBoost'
]

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

n_features = 10 #Number of top features to parse

for name in names:

    print("{0} top features:\n".format(name))

    try:
        f = open('{0}_features.txt'.format(name), 'w')

    except FileNotFoundError:
        f = open('{0}_features.txt'.format(name), 'x')

    f.write("MODEL: {0}\n".format(name))
    f.write("=====================================\n")

    df = pd.read_csv('{0}_results.csv'.format(name))
    dropped = list(df.iloc[5])

    prev_appended = []

    for i in reversed(dropped[int(-(n_features/2)):]): #Only parse features as given above
        
        l = name_separator(i)
        
        #Each new string includes previously parsed features, this removes them
        l = [j for j in l if (j not in prev_appended)]

        prev_appended += l

        print(">>> ", l)

        f.write("\n{0}\n".format(l))
        
    print("\n====================================\n")

    f.close()



