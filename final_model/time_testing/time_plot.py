import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('results.csv', delimiter=",", unpack=True)

datamap = {
        1: 'Accuracy',
        2: 'Precision',
        3: 'Recall',
        4: 'f1'
}

for i in range(1,5):
    plt.figure()
    
    plt.plot(data[:,0], data[:,0+i], "--", label="Non-balanced dataset")

    plt.title('XGBoost {0} from model trained on data balanced to 50:50\nmalware:benign, from 2000-20XX, tested on data from 2019-21'.format(datamap[i]))
    plt.xlabel('End year of training set 2000-20XX')
    plt.ylabel('Model {0}'.format(datamap[i]))
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig('graphs/{0}_balances.png'.format(datamap[i]))

