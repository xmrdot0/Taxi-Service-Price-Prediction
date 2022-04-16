import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error



from sklearn.linear_model import LinearRegression

taxi_rides = pd.read_csv('taxi-rides.csv')
weather =  pd.read_csv('weather.csv')

# Cleaning and merging data

# Removing rows that has no price
taxi_rides = taxi_rides.dropna(axis = 0).reset_index(drop = True)

# Changing values of days that has NA rain to 0
weather = weather.fillna(0)

# Creating an average weather dataframe based on location
average_weather = weather.groupby('location').mean().reset_index(drop=False)
average_weather = average_weather.drop('time_stamp', axis=1)

# Rename the columns to fit in the main data dataframe
source_weather = average_weather.rename(columns= {
    'location': 'source',
    'temp':'source-temp',
    'clouds':'source-clouds',
    'pressure':'source-pressure',
    'rain':'source-rain',
    'humidity':'source-humidity',
    'wind':'source-wind'
})
destination_weather = average_weather.rename(columns= {
    'location': 'destination',
    'temp':'destination-temp',
    'clouds':'destination-clouds',
    'pressure':'destination-pressure',
    'rain':'destination-rain',
    'humidity':'destination-humidity',
    'wind':'destination-wind'
})

# Merge the original data with the weather for both source and destination
data = taxi_rides.merge(source_weather, on ='source')
data = taxi_rides.merge(destination_weather, on ='destination')


#########################################################################

# Preprocessing data

def preprocess_inputs(X):
    X = X.copy()

    # Drop unused columns
    X = X.drop(['id'], axis = 1)

    # Binary-encode cab_type column
    X['cab_type'] = X['cab_type'].replace(
        {
            'Uber':0,
            'Lyft':1 
        }
    )

    # Encoding categorical columns
    for c in ['destination', 'source', 'product_id', 'name']:
        lbl = LabelEncoder()
        lbl.fit(list(X[c].values))
        X[c] = lbl.transform(list(X[c].values))

    y = X['price']
    x = X.drop('price', axis=1)

    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.7, shuffle = True, random_state = 1)

    return X_train, X_test, Y_train, Y_test


X_train, X_test, Y_train, Y_test = preprocess_inputs(data)


# Training

LR = LinearRegression()

LR.fit(X_train,Y_train)

y_prediction =  LR.predict(X_test)


print(mean_squared_error(Y_test, y_prediction))

