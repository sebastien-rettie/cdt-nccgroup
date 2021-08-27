import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('tripleyear_results.csv',unpack=True,delimiter=",")

years = data[:,0]
accs = data[:,1]
sizes = data[:,2]
print(years,accs,sizes)

plt.figure(1)
plt.xlabel("Training dataset size (number of files)")
plt.ylabel("Random Forest Accuracy")
plt.title("Random Forest accuracy on 2019-22 test files trained on\ndifferently sized triple-year (consecutive) datasets from 1970-2018")
plt.scatter(sizes,accs,color='k')
plt.grid(True)
plt.savefig('graphs/triple_size_accuracy.png')

plt.figure(2)
plt.xlabel("Datestamp end-year of triple-year training files")
plt.ylabel("Random Forest Accuracy")
plt.title("Random Forest accuracy on 2019-22 test files trained on\ntriple-year (consecutive) datasets from 1970-2018")
plt.plot(years,accs,'k-')
plt.grid(True)
plt.savefig('graphs/triple_year_accuracy.png')
