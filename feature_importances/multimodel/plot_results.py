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
            'f1': df_f1
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

        current_feature_names = [] #For plotting purposes
        plot_data = []

        for i in final[:10]:
            print('{0}: {1}'.format(feature_list[i[0]], i[1]))
            current_feature_names.append(feature_list[i[0]])
            plot_data.append(i[1])

        print("\n10 least important features (possible hindrances):")

        for i in final[-10:]:
            print('{0}: {1}'.format(feature_list[i[0]], i[1]))

        plt.figure()
        plt.xlabel('Feature Name')
        plt.ylabel('Model {0} reduction from removal'.format(key))
        plt.title('{0} top 10 feature importances - {1} metric'.format(name, key))
        plt.bar(np.arange(1,11,1), plot_data, tick_label=current_feature_names)
        plt.xticks(fontsize=10, rotation='vertical')
        plt.savefig('{0}_graphs/{1}_top10.png'.format(name,key))

    if name != names[-1]:
        print('\n\nThe next model is ready to show.')
        msg = str(input('Press enter to show the next set of results. '))
        continue
