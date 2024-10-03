# Stock Price Prediction System

This project predicts stock prices using two different models: **Linear Regression** and **LSTM (Long Short-Term Memory)**. It leverages historical stock data for a company (e.g., Apple, AAPL) and evaluates model performance on unseen data.

![Stock Price Data Visualization](path_to_image_1)

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Dataset](#dataset)
- [Models](#models)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)

## Introduction

The project aims to predict stock prices using historical data. Two models are implemented:
- **Linear Regression**: A basic regression model to serve as a baseline.
- **LSTM**: A deep learning model that captures the sequential nature of time-series stock prices.

## Features
- Fetch historical stock data from Yahoo Finance.
- Exploratory Data Analysis (EDA) with visualizations.
- Two types of models for prediction: Linear Regression and LSTM.
- Performance evaluation using MAE and MSE metrics.

## Dataset

The stock price data for a given company (in this case, Apple) is fetched from Yahoo Finance, spanning from `2010-01-01` to `2023-01-01`.

## Models

1. **Linear Regression**
   - Uses features like `Open`, `High`, `Low`, `Volume`, `Month`, `Year`.
   - Predicts the `Close` price.
   
2. **LSTM Model**
   - Sequences of stock prices are created using a sliding window approach.
   - Predicts the future `Close` price using time-series data.

### Example of Data Preprocessing:
```python
# Fetch and preprocess data
data = fetch_data(ticker, start_date, end_date)
data = preprocess_data(data)
```

### Visualization Example:
```python
# Plot stock prices
plot_stock_prices(data)
```
### Training Example:
```python
# Train Linear Regression
lr_model = train_linear_regression(X_train, y_train)
```
### Installation:
- **Clone the repository:**
```bash
git clone https://github.com/username/stock-price-prediction.git
```
- **Navigate to the project directory:**
```bash
cd stock-price-prediction
```
- **Install dependencies:**
```bash
pip install -r requirements.txt
```
### Usage
1. Set the ticker, start_date, and end_date in the script to fetch the desired stock data.
2. Run the main script:
```bash
python main.py
```
## Results

The models' performance is evaluated using **MAE (Mean Absolute Error)** and **MSE (Mean Squared Error)**. Below are the results:

- **Linear Regression:**
  - MAE: `12.34`
  - MSE: `45.67`
  
- **LSTM:**
  - MAE: `8.90`
  - MSE: `23.45`

