import time
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from pre_processing import *
from sklearn.linear_model import LinearRegression

start_time = time.time()
taxi_rides = pd.read_csv('taxi-rides.csv')
weather = pd.read_csv('weather.csv')

# Cleaning and merging data

# Removing rows that have no price
taxi_rides = taxi_rides.dropna(axis = 0).reset_index(drop = True)

# Changing values of days that has NA rain to 0
weather = weather.fillna(0)

# Creating an average weather dataframe based on location
average_weather = weather.groupby('location').mean().reset_index(drop=False)

average_weather = average_weather.drop('time_stamp', axis=1)

average_weather.to_csv('average_weather.csv', index=False)

# pickle.dump( average_weather, open( "average_weather.sav", "wb" ) )


data = merge_data(average_weather, taxi_rides)


# Preprocessing data


X_train, X_test, Y_train, Y_test = preprocess_inputs(data)


#########################################################################

# Training

LR = LinearRegression()

LR.fit(X_train,Y_train)

pickle.dump( LR, open( "multivariable.sav", "wb" ) )

#LR = pickle.load( open( "multivariants.sav", "rb" ) )

y_prediction = LR.predict(X_test)

#print('Co-efficient of linear regression:',LR.coef_)
print("mean square error of test data set:", mean_squared_error(Y_test, y_prediction))

true_value = np.asarray(Y_test)[0]
predicted_value = y_prediction[0]
print('True value  in the test set  : ' + str(true_value))
print('Predicted value for the first element in the test set : ' + str(predicted_value))
print("R2 Score : ", LR.score(X_test, Y_test))

#print("time taken %s seconds ---" % (time.time() - start_time))
