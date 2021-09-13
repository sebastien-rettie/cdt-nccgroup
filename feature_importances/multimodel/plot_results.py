import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

names = [
    "NearestNeighbors",
    "SGDSVM",
    "DecisionTree",
    "RandomForest",
    "XGBoost",
    "NeuralNet",
    "AdaBoost",
]

time_averages = []
time_stds = []

for name in names:
    print('\n#############################')
    print("MODEL: {0}".format(name))
    print('#############################\n') #Why not be decorative 

    df = pd.read_csv('{0}_features.csv'.format(name))
    
    feature_list = list(df.columns)

    df_accuracy = np.array(df.iloc[0])
    df_precision = np.array(df.iloc[1])
    df_recall = np.array(df.iloc[2])
    df_f1 = np.array(df.iloc[3])
    df_time = np.array(df.iloc[4])
    
    data_map = {
            'Accuracy': df_accuracy,
            'Precision': df_precision,
            'Recall': df_recall,
            'f1': df_f1,
            'Time': df_time
    }

    for key in data_map.keys():
        try:
            plt.clf() #Clear previous figure
        except:
            pass

        print('\n{0}'.format(key))
        print('===========================\n')

        data = data_map[key][-1] - data_map[key][:-1]
    
        final = []
        index = 0
        while index < len(data):
            final.append((index, data[index]))
            index += 1

        final.sort(key=lambda x: x[1], reverse=True)
    
        print("Top 10 feature importances:")

        top_feature_names = [] #For plotting purposes
        top_plot_data = []
        bottom_feature_names = []
        bottom_plot_data = []

        for i in final[:10]:
            print('{0}: {1}'.format(feature_list[i[0]], i[1]))
            top_feature_names.append(feature_list[i[0]])
            top_plot_data.append(i[1])

        print("\n10 least important features (possible hindrances):")

        for i in final[-10:]:
            print('{0}: {1}'.format(feature_list[i[0]], i[1]))
            bottom_feature_names.append(feature_list[i[0]])
            bottom_plot_data.append(i[1])

        ### Top features ###

        plt.figure(num='{0}{1}top'.format(name,key))
        plt.xlabel('Feature Name')

        if key != 'Time': #All metrics except time
            plt.ylabel('Model {0} % reduction from removal'.format(key))
            plt.bar(np.arange(1,11,1), 100*np.array(top_plot_data), tick_label=top_feature_names)
        else: #if time is metric, don't multiply data by 100 (not %), add units in axis
            plt.ylabel('Model training time reduction from removal (s)')
            plt.bar(np.arange(1,11,1), np.array(top_plot_data), tick_label=top_feature_names)

        plt.title('{0} top 10 feature importances - {1} metric'.format(name, key))
        plt.xticks(fontsize=10, rotation='vertical')
        plt.tight_layout()
        plt.savefig('{0}_graphs/{1}_top10.png'.format(name,key))

        ### Bottom features ###

        plt.figure(num='{0}{1}bottom'.format(name,key))
        plt.xlabel('Feature Name')
        plt.ylabel('Model {0} % reduction from removal'.format(key))

        if key != 'Time': #All metrics except time
            plt.ylabel('Model {0} % reduction from removal'.format(key))
            plt.bar(np.arange(1,11,1), 100*np.array(bottom_plot_data), tick_label=bottom_feature_names)
        else: #if time is metric, don't multiply data by 100 (not %), add units in axis
            plt.ylabel('Model training time reduction from removal (s)')
            plt.bar(np.arange(1,11,1), np.array(bottom_plot_data), tick_label=bottom_feature_names)

        plt.title('{0} bottom 10 feature importances - {1} metric'.format(name, key))
        plt.xticks(fontsize=10, rotation='vertical')
        plt.tight_layout()
        plt.savefig('{0}_graphs/{1}_bottom10.png'.format(name,key))

    if name != names[-1]:
        print('\n\nThe next model is ready to show.')
        msg = str(input('Press enter to show the next set of results. '))
        continue
