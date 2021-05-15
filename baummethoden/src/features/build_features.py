import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class Preprocessor:
    def __init__(self, data):
    #Shuffle data
        data = data.sample(frac=1)

        self.X = data.drop(['class'], axis = 1)
        self.y = data['class']
        
        # hier werden daten gesplittet
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X,self.y, test_size=0.2, random_state=14)
        # shapes of our data splits
        print('train Dataset',self.x_train.shape) 
        print('test Dataset',self.x_test.shape)
        print('train lables',self.y_train.shape)
        #print(self.y_test.shape)
                     
        # ab hier hab ich jetzt den scaler fertig
        
        #Statistics
        #Get the absolute number of how many instances in our data belong to class Positive
        self.count_fake = len(data.loc[data['class']== 1.0])
        print('Fake bills abolute: ' +str(self.count_fake))

        #Get the absolute number of how many instances in our data belong to class one
        self.count_real = len(data.loc[data['class']== 0.0])
        print('Real bills absolute: ' + str(self.count_real))


  
    def get_data(self):
        return self.x_train, self.x_test, self.y_train, self.y_test, self.X, self.y

    def statistica(self):
        
        #Get the relative number of how many instances in our data belong to class zero
        percentage_fake = round(self.count_fake/(self.count_real + self.count_fake),4)*100
        print('Fake bills in percent: ' + str(round(percentage_fake,3)))
        
        #Get the relative number of how many instances in our data belong to class one
        percentage_real = round(self.count_real/(self.count_fake + self.count_real),4)*100
        print('Real bills in percent: ' + str(round(percentage_real,3)))

        return self.count_fake, self.count_real, percentage_fake, percentage_real
