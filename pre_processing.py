import pandas as pd
from sklearn.model_selection import train_test_split

# Preprocessing data


def merge_data(weather_data, taxi_rides_data):

    # Rename the columns to fit in the main data dataframe
    source_weather = weather_data.rename(columns={
        'location': 'source',
        'temp': 'source-temp',
        'clouds': 'source-clouds',
        'pressure': 'source-pressure',
        'rain': 'source-rain',
        'humidity': 'source-humidity',
        'wind': 'source-wind'
    })
    destination_weather = weather_data.rename(columns={
        'location': 'destination',
        'temp': 'destination-temp',
        'clouds': 'destination-clouds',
        'pressure': 'destination-pressure',
        'rain': 'destination-rain',
        'humidity': 'destination-humidity',
        'wind': 'destination-wind'
    })

    # Merge the original data with the weather for both source and destination
    data = taxi_rides_data.merge(source_weather, on='source')
    data = taxi_rides_data.merge(destination_weather, on='destination')
    return data


def onehot_encode(df, column, prefix):
    df = df.copy()
    dummies = pd.get_dummies(df[column], prefix=prefix)
    df = pd.concat([df,dummies], axis=1)
    df = df.drop(column,axis=1)
    return df


def preprocess_inputs(X):
    X = X.copy()

    # Drop unused columns
    X = X.drop(['id'], axis=1)

    # Binary-encode cab_type column
    X['cab_type'] = X['cab_type'].replace(
        {
            'Uber':0,
            'Lyft':1
        }
    )

    # Label Encoding categorical columns
    # for c in ['destination', 'source', 'product_id', 'name']:
    #     lbl = LabelEncoder()
    #     lbl.fit(list(X[c].values))
    #     X[c] = lbl.transform(list(X[c].values))

    # One Hot Encoding categorical columns
    for column, prefix in [('destination', 'dest'), ('source','src'), ('product_id','pid'), ('name','nm')]:
        X = onehot_encode(X, column=column, prefix=prefix)

    y = X['price']
    x = X.drop('price', axis=1)

    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3, shuffle = True, random_state = 1)

    return X_train, X_test, Y_train, Y_test
