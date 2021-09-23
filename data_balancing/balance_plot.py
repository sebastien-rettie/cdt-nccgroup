import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('balance_results.csv', delimiter=",", unpack=True)

datamap = {
        1: 'Accuracy',
        2: 'Precision',
        3: 'Recall',
        4: 'f1'
}

fractions = [0.25,0.5,0.75]

for i in range(1,5):
    fraction_index = 0

    for frac in fractions:
        plt.figure()
        plt.plot(100*data[:,0], data[:,(5*fraction_index)+i], label="Test dataset: {0}% malware".format(100*frac))
        plt.title('XGBoost {0} from model trained on various balanced data,\ntested on data balanced to 25/50/75% malware'.format(datamap[i],100*frac))
        plt.xlabel('Training dataset malware balance (%)')
        plt.ylabel('Model {0}'.format(datamap[i]))
        plt.legend(loc='best')
        plt.grid(True)
        plt.savefig('graphs/{0}_balance_curve.png'.format(datamap[i]))

        fraction_index += 1

