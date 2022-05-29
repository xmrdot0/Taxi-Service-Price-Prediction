import time
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures
from pre_processing import *
import numpy as np
import pickle
start_time = time.time()
taxi_rides = pd.read_csv('taxi-rides.csv')

weather = pd.read_csv('weather.csv')

# Cleaning and merging data

# Removing rows that has no price
taxi_rides = taxi_rides.dropna(axis = 0).reset_index(drop = True)

# Changing values of days that has NA rain to 0
weather = weather.fillna(0)

# Creating an average weather dataframe based on location
average_weather = weather.groupby('location').mean().reset_index(drop=False)
average_weather = average_weather.drop('time_stamp', axis=1)

# Merge the original data with the weather for both source and destination
data = merge_data(average_weather, taxi_rides)


# Feature Selection but not used in filtering data that
# it causes a big mean square error and low accuracy
# So we use it only to plot a simple graph describing the correlation
# Get the correlation between the features
corr = data.corr()
# Top 50% Correlation training features with the Value
top_feature = corr.index[abs(corr['price'])>0.2]
# Correlation plot
# plt.subplots(figsize=(12, 8))
# top_corr = data[top_feature].corr()
# #sns.heatmap(top_corr, annot=True)
# plt.show()

# Preprocessing data

X_train, X_test, Y_train, Y_test = preprocess_inputs(data)

#########################################################################

# training

poly_features = PolynomialFeatures(degree=2)

# transforms the existing features to higher degree features.

X_train_poly = poly_features.fit_transform(X_train)

pickle.dump( poly_features, open( "Polynomial.sav", "wb" ) )


# fit the transformed features to Linear Regression
poly_model = linear_model.LinearRegression()
poly_model.fit(X_train_poly, Y_train)

# predicting on training data-set
y_train_predicted = poly_model.predict(X_train_poly)

# predicting on test data-set
prediction = poly_model.predict(poly_features.fit_transform(X_test))

# output

#print('Co-efficient of linear regression',poly_model.coef_)

print('Mean Square Error of test', metrics.mean_squared_error(Y_test, prediction))
print('Mean Square Error of training', metrics.mean_squared_error(Y_train, y_train_predicted))

true_value=np.asarray(Y_test)[0]
predicted_value=prediction[0]
print('True value  in the test set  : ' + str(true_value))
print('Predicted value for the first element in the test set : ' + str(predicted_value))

print("R2 Score : ", poly_model.score(poly_features.fit_transform(X_test), Y_test))
#print("time taken %s seconds ---" % (time.time() - start_time))
