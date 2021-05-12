# cdt-nccgroup
2021 CDT group project with NCC Group.

Binary and malign samples have been gathered, checked for duplicates and placed on the AWS server. 
(Duplication check was performed with a linux tool called fdupes)

This repository holds scripts for extracting characteristics from the raw binaries, converting into features and training machine learning models. 

Some can be run locally, but most require more computing power than a local workstation is capable of and should be run on the cluster. 

We use three possible methods for extracting characteristics from the executables:
- Bytes n-grams
- Call graphs (API and opcode)
- PE header fields

We then test multiple ML models before creating an ensemble based on a model from each feature extraction method.  

![Workflow](https://github.com/sebastien-rettie/cdt-nccgroup/blob/main/workflow.png?raw=true)

<h3>PE Header Workflow</h3>
(If using preprocessed PE data, skip to 4)
pefile, sklearn and hyperopt are among several required packages.
To install the required packages in conda, download ML_PE.yml and run <code>conda env create --file ML_PE.yml</code> , then <code>conda activate ML_PE</code> .

1. To extract features, place extract-pe.py (in feature_extraction/pe) in a folder benign or malware samples and run.
This generates a .csv file of extracted header information for all binaries, either processed_malware.csv or process_benign.csv
Binaries take up a lot of memory so it is best to do this in batches.
2. To combine the .csv files, use the concat_files function in preprocessing.py (NOT linux cat which will produce mangled results) There is an example at the end of the script. This is a good point to perform the train/test split.
3. preprocessing.py also contains the workflow for dropping NaNs and ensuring train/test have same number of columns (important for one-hot encoding). This outputs test.csv and train.csv 

4. The models (modelname.py in pe/ml_models) import the preprocessed .csv data and divide it into input and target. (X and y)
For a given dataset 1H-encoding, feature scaling and selection only need to be run once - then their output can be saved. This saves refitting the transformers each time. decisiontree.py fits the transformers and saves them as .pickle files.
Run this once.
(Note: It's probably better to save the transformed data here instead of constantly retransforming it, feel free to implement that.)
5.  Other scripts in the folder import the fitted transformers to train different ML models and report on their performance. Run these and set performance_report = True to calculate accuracy score among other metrics.
6. Some hyperparameter tuning is performed in these scripts. Set previously_tuned = False to perform tuning.
However a better option is modelname_tuning.py which uses the hyperopt package. Hyperopt is the recommended method for fast and effective tuning so run these to find the optimal hyperparameters for a given model.
7. The modelname.py scripts import from plotting.py to make useful plots such as roc, validation and learning curves. Set plot_validate_params = True to validation curves and check for overfitting.
8. multimodel.py and ensemble.py train and compare multiple models and ensembles  resepectively. Run these to directly compare models and the performance of ensembles based on them. 
