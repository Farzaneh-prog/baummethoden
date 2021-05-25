import sys
from sklearn.tree import DecisionTreeClassifier
from subprocess import check_call, check_output

import pandas as pd
import pickle as pi

current_dir = 'baummethoden'
#sys.path.append('../../src/features/')
sys.path.append('../baummethoden/src/features/')
import build_features
from build_features import *
direct = check_output('pwd')
print(direct)
sys.path.append('../../src/visualization/')
#check_call('pwd')
import visualize
from visualize import *


#List with attribute names (it is optional to do this but it gives a better understanding of the data for a human reader)
attribute_names = ['variance_wavelet_transformed_image', 'skewness_wavelet_transformed_image', 'curtosis_wavelet_transformed_image', 'entropy_image', 'class']

#Read csv-file
#data = pd.read_csv('/home/farzaneh/DataScientist/LearnPython/Baummethoden/baummethoden/data/processed/data_banknote_authentication.csv', names=attribute_names)
data = pd.read_csv('../../../'+current_dir+'/data/processed/data_banknote_authentication.csv', names=attribute_names)

#preparing data for modelling
preprocessor = Preprocessor(data)
x_train, x_test, y_train, y_test, X, y = preprocessor.get_data()

#Data Statistics
count_fake, count_real, percentage_fake, percentage_real = preprocessor.statistica()

#Create a classifier object 
classifier = DecisionTreeClassifier() 

#Classfier builds Decision Tree with training data
classifier = classifier.fit(x_train, y_train) 

#Shows importances of the attributes according to our model 
classifier.feature_importances_


#graphical output
Baum_tree(classifier, attribute_names)

#save in Pickle file
path_start = os.getcwd()
pathr = os.path.dirname(os.getcwd())+'/../models'
os.chdir(pathr)
file_name = "classification_model.pickle"
fill = open(file_name,'wb')     #allow to Write the file in a Binary format
pi.dump(classifier,fill)
fill.close()

#change to the start working directory
os.chdir(path_start)