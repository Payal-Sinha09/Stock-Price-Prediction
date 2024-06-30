import pandas as pd
from src.data_fetch import fetch_data
from src.data_preprocessing import preprocess_data, scale_data
from src.eda import plot_stock_prices
from src.model import train_linear_regression, evaluate_model, build_lstm_model, train_lstm_model
from src.utils import create_sequences

# Parameters
ticker = 'AAPL'
start_date = '2010-01-01'
end_date = '2023-01-01'
sequence_length = 60

# Fetch and preprocess data
data = fetch_data(ticker, start_date, end_date)
data = preprocess_data(data)

# Exploratory Data Analysis
plot_stock_prices(data)

# Prepare data for Linear Regression
features = ['Open', 'High', 'Low', 'Volume', 'Month', 'Year']
target = 'Close'
X = data[features]
y = data[target]
X_scaled, scaler = scale_data(data, features)

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train and evaluate Linear Regression model
lr_model = train_linear_regression(X_train, y_train)
mae_lr, mse_lr = evaluate_model(lr_model, X_test, y_test)
print(f'Linear Regression MAE: {mae_lr}')
print(f'Linear Regression MSE: {mse_lr}')

# Prepare data for LSTM
X_lstm, y_lstm = create_sequences(data['Close'].values, sequence_length)
X_train_lstm, X_test_lstm = X_lstm[:int(X_lstm.shape[0]*0.8)], X_lstm[int(X_lstm.shape[0]*0.8):]
y_train_lstm, y_test_lstm = y_lstm[:int(y_lstm.shape[0]*0.8)], y_lstm[int(y_lstm.shape[0]*0.8):]
X_train_lstm = X_train_lstm.reshape((X_train_lstm.shape[0], X_train_lstm.shape[1], 1))
X_test_lstm = X_test_lstm.reshape((X_test_lstm.shape[0], X_test_lstm.shape[1], 1))

# Build and train LSTM model
lstm_model = build_lstm_model(sequence_length)
lstm_model = train_lstm_model(lstm_model, X_train_lstm, y_train_lstm)

# Evaluate LSTM model
y_pred_lstm = lstm_model.predict(X_test_lstm)
mae_lstm = evaluate_model(lstm_model, X_test_lstm, y_test_lstm)
print(f'LSTM MAE: {mae_lstm}')
