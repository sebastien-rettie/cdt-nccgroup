import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('results.csv', delimiter=",", unpack=True)

#######################################
#--------------CHI FIT----------------#

def chifit(x,y):
    order = 2 #polynomial order of fit
    p, resid, rank, sing, rcond = np.polyfit(x,y,order,full=True)

    npoints = len(x)
    ncoeffs = len(p)
    dof = npoints - ncoeffs   #Degrees of freedom
    print('Fit degrees of freedom: ',dof)

    chi = (resid/dof)[0]
    print('Reduced chi^2: ',chi)

    error = np.sqrt(resid[0]/dof)
    print('Estimated error:', error)

    poly = np.poly1d(p)
    ypoints = poly(x)                          #y value array using poly1d
    
    return (x, ypoints, chi, error)
#-------------------------------------#
#######################################

datamap = {
        1: 'Accuracy',
        2: 'Precision',
        3: 'Recall',
        4: 'f1'
}

for i in range(1,5):
    plt.figure()
    plt.plot(data[:,0], data[:,i], "--", label="Collected data")
    plt.title('XGBoost {0} from model trained on data from\n 2000-20XX, tested on data from 2019-21'.format(datamap[i]))
    plt.xlabel('End year of training set 2000-20XX')
    plt.ylabel('Model {0}'.format(datamap[i]))
    plt.grid(True)
    plt.savefig('graphs/{0}'.format(datamap[i]))

    (x, y, chi, err) = chifit(data[:,0], data[:,i])
    plt.plot(x, y, "k-", label="Fitted polynomial $\chi^2_r =$ {0:0.3f}".format(chi))
    plt.legend(loc='best')
    plt.annotate('Estimated {0} Error: {1:0.3f}'.format(datamap[i],err), xy=(0.05, 0.78), xycoords='axes fraction')
    plt.savefig('graphs/{0}_chifit'.format(datamap[i]))
