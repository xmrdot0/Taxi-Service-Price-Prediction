import pickle

import pandas as pd
from pre_processing import *

multivariable = pickle.load(open( "multivariable.sav", "rb" ))

taxi_rides = pd.read_csv('taxi-rides.csv')
weather = pd.read_csv('weather.csv')

# Removing rows that have no price
taxi_rides = taxi_rides.dropna(axis = 0, subset=['price']).reset_index(drop = True)

# Changing values of days that has NA rain to 0
weather = weather.fillna(0)

# Creating an average weather dataframe based on location
average_weather = weather.groupby('location').mean().reset_index(drop=False)


average_weather = average_weather.drop('time_stamp', axis=1)


data = merge_data(average_weather, taxi_rides)
