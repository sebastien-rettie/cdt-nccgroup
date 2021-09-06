import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

names = [
    "DecisionTree",
    "RandomForest",
    "XGBoost",
    "AdaBoost",
]

for name in names:

    df = pd.read_csv('{0}_results.csv'.format(name))

    features = np.arange(float(list(df.columns)[0]),float(list(df.columns)[-1])+2,2)
    features_corrected = np.array([240]*len(df.columns) - features)

    acc_array = np.array(df.iloc[0])
    prec_array = np.array(df.iloc[1])
    rec_array = np.array(df.iloc[2])
    f1_array = np.array(df.iloc[3])
    time_array = np.array(df.iloc[4])

    arr_map = {
            'Accuracy': acc_array,
            'Precision': prec_array,
            'Recall': rec_array,
            'f1': f1_array,
            'Training time': time_array
    }

    for key in arr_map.keys():

        plt.figure('{0}{1}'.format(name,key))
        plt.plot(features_corrected, arr_map[key])
        plt.title('{0} - {1} over reduced features'.format(name,key))
        plt.xlabel('Number of input features')
        
        if key == 'Training time':
            plt.ylabel('{0} of model (s)'.format(key))
        else:
            plt.ylabel('{0} of model'.format(key))

        plt.grid(True)
        plt.savefig('{0}_graphs/{1}.png'.format(name,key))

