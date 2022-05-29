import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pickle
from pre_processing_2 import *

Weather_data = pd.read_csv('weather.csv')
rides_data = pd.read_csv('taxi-rides-classification.csv')


rides_data.drop(rides_data[rides_data['RideCategory'] == 'unknown'].index, axis = 0, inplace=True)


Weather_data = Weather_data.fillna(0)

# Creating an average weather dataframe based on location
average_weather = Weather_data.groupby('location').mean().reset_index(drop=False)
average_weather = average_weather.drop('time_stamp', axis=1)


data = merge_data(average_weather, rides_data)

print("Merge done")


X_train, X_test, Y_train, Y_test= preprocess_inputs(data)

print("Preproccessing done")
 

model3 = DecisionTreeClassifier()

model3.fit(X_train, Y_train)


pickle.dump( model3, open( "DecisionTreeClassifier.sav", "wb" ) )


#calculcate Accuracy on Training data

X_train_prediction2 = model3.predict(X_train)

training_data_accuracy2 = accuracy_score(X_train_prediction2, Y_train)



print('Accuracy on Training data_SVC : ', training_data_accuracy2)


#calculcate Accuracy on Test data
X_test_prediction1 = model3.predict(X_test)
testing_data_accuracy1 = accuracy_score(X_test_prediction1, Y_test)


print('Accuracy on Testing data_SVC : ', testing_data_accuracy1)