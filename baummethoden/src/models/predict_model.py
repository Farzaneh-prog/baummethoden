import os
import pandas as pd
import pickle as pi

#save in Pickle file
path_start = os.getcwd()
pathr = os.path.dirname(os.getcwd())+'/../models'
os.chdir(pathr)
file_name = "classification_model.pickle"
fill = open(file_name,'rb')     #read only file in a Binary format
classifier = pi.load(fill)
fill.close()

#change to the start working directory
os.chdir(path_start)

#List with attribute names (it is optional to do this but it gives a better understanding of the data for a human reader)
attribute_names = ['variance_wavelet_transformed_image', 'skewness_wavelet_transformed_image', 'curtosis_wavelet_transformed_image', 'entropy_image']

pathread = os.path.dirname(os.getcwd())+'/../data/external/Predict.csv'
#Read csv-file
data = pd.read_csv(pathread, names=attribute_names)

#Get predicted values from test data 
y_pred = classifier.predict(data) 
#print(y_pred)

erg = pd.DataFrame(y_pred, columns = ['prediction'])
ergebnis = pd.concat([data,erg],axis=1)
print(ergebnis)

pathsave = os.path.dirname(os.getcwd())+'/../reports/prediction.xlsx'
ergebnis.to_excel(pathsave, sheet_name = 'Sheet_name_1')