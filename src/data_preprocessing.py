import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(data):
    data = data.dropna()
    data['Date'] = pd.to_datetime(data.index)
    data.set_index('Date', inplace=True)
    data['Month'] = data.index.month
    data['Year'] = data.index.year
    return data

def scale_data(data, features):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[features])
    return scaled_data, scaler
