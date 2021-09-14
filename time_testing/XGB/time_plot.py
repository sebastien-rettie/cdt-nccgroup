import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('results.csv', delimiter=",", unpack=True)

datamap = {
        1: 'Accuracy',
        2: 'Precision',
        3: 'Recall',
        4: 'f1'
}

fractions = [0.25,0.5,0.75,'N/A']

for i in range(1,5):
    plt.figure()
    
    fraction_index = 0

    for frac in fractions:
        if frac != 'N/A':
            plt.plot(data[:,0], data[:,(5*fraction_index)+i], "--", label="{0:0.0f}% malware balance".format(100*frac))
        else:
            plt.plot(data[:,0], data[:,(5*fraction_index)+i], "-", label="Non-balanced dataset")

        plt.title('XGBoost {0} from model trained on various balanced data\nfrom 2000-20XX, tested on data from 2019-21'.format(datamap[i]))
        plt.xlabel('End year of training set 2000-20XX')
        plt.ylabel('Model {0}'.format(datamap[i]))
        plt.legend(loc='best')
        plt.grid(True)
        plt.savefig('graphs/{0}_balances.png'.format(datamap[i]))

        fraction_index += 1

