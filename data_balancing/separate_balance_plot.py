import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('separate_balance_results.csv', delimiter=",", unpack=True)

datamap = {
        1: 'Accuracy',
        2: 'Precision',
        3: 'Recall',
        4: 'f1'
}

targets = ['benign','malware']

print(data)

for i in range(1,5):
    target_index = 0

    for target in targets:
        plt.figure()
        plt.plot(100*data[:,0], data[:,(5*target_index)+i])
        plt.title('XGBoost {0} from model trained on data with\nincreasing amounts of {1}, unbalanced test set used'.format(datamap[i],target.capitalize()))
        plt.xlabel('Percentage of {0} statistics used (%)'.format(target))
        plt.ylabel('Model {0}'.format(datamap[i]))
        plt.grid(True)
        plt.savefig('separate_balance_graphs/{0}_{1}_balance_curve.png'.format(target, datamap[i]))

        target_index += 1

