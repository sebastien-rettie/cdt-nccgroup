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

datasets = []
for name in names:
    datasets.append((name, np.loadtxt('model_results/triple/{0}_tripleyear_results.csv'.format(name),unpack=True,delimiter=",")))

#######################################
#--------------CHI FIT----------------#

#Build full dataset from every separate one
n_years = len(datasets[0][1][:,0])

full_accs = np.zeros((n_years, len(datasets)))

#group model accuracies by year in matrix
index = 0 #iteration counter through each model
for d in datasets:
    data = d[1]
    
    second_index = 0 #iterate through years
    while second_index < len(data[:,0]): #i less than length of year range in matrix
        
        full_accs[second_index, index] = data[second_index, 1] #append to index[year, model] in full_accs
        second_index += 1

    index += 1

x = []
y = []

i = 0
while i < len(full_accs):
    for j in full_accs[i]:
        y.append(j)
        x.append(i+1999) #append year to x
    i+=1

#at this point x,y are full sets of data, these look like
#x = [1999, 1999, ..., 2000, 2000, ...]
#y = [1999 model 1 acc, 1999 model 2 acc, ..., 2000 model 1 acc, 2000 model 2 acc, ...]

order = 2 #polynomial order of fit
p, resid, rank, sing, rcond = np.polyfit(x,y,order,full=True)

npoints = len(x)
ncoeffs = len(p)
dof = npoints - ncoeffs   #Degrees of freedom
print('Fit degrees of freedom: ',dof)

chi = (resid/dof)[0]
print('Reduced chi^2: ',chi)

poly = np.poly1d(p)
xpoints = np.linspace(x[0],x[len(x)-1],n_years)   #x value array
ypoints = poly(xpoints)                          #y value array using poly1d
#-------------------------------------#
#######################################

plt.figure(1)
plt.xlabel("Datestamp end-year of triple-year training files")
plt.ylabel("Model Accuracy")
plt.title("Various model accuracy on 2019-21 test files trained on\ntriple-year (consecutive) datasets from 1999-2018")
plt.grid(True)

for d in datasets:
    data = d[1]
    years = data[:,0]
    accs = data[:,1]
    sizes = data[:,2]

    plt.scatter(sizes,accs,label=d[0])

plt.legend(loc="best")
plt.savefig('graphs/tripleyear_size_accuracy.png')

plt.figure(2)
plt.xlabel("Datestamp end-year of triple-year training files")
plt.ylabel("Model Accuracy")
plt.title("Various model accuracy on 2019-21 test files trained on\ntriple-year (consecutive) datasets from 1999-2018")
plt.grid(True)

for d in datasets:
    data = d[1]
    years = data[:,0]
    accs = data[:,1]
    sizes = data[:,2]
    
    plt.plot(years,accs,'--',label=d[0])

plt.legend(loc="best")
plt.savefig('graphs/tripleyear_accuracy_plot.png')

plt.plot(xpoints, ypoints, "k-", label="2nd order fit, $\chi_r^2 =${0:0.3f}".format(chi))
plt.legend(loc="best")
plt.savefig('graphs/tripleyear_accuracy_chifit.png')
