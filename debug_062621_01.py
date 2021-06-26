
# %% [markdown]
# # Predictions from all data, no tests to verify, start below

# %%
# import, prep data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sqlite3
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
import datetime
import os
from sklearn.metrics import mean_squared_error
test_symbols = ["CTXR", "IBIO", "MNKD", "IMGN", "ATOS", "SPPI"]
model_settings = {'epochs': 10, 'batch_size': 100, 'train_test_ratio': 0.7, 'hidden_layers': 3, 'units': 100}

# Choose what Symbol you would like to predict:
symbol = 'CTXR'     # change to whatever symbol in your dataframe you'd like to attempt to predict.
start_date = "2020-01-01"
current_date = datetime.datetime.now().strftime("%Y-%m-%d")
current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_settings['start_date'] = start_date
model_settings['end_date'] = current_date

# next cell
conn = sqlite3.connect('stockPrediction_05262021.db')

symbol_info = conn.execute(f"SELECT sector_id, industry_id FROM stock WHERE stock_symbol = \"{symbol}\";")
symbol_info = symbol_info.fetchall()
sector_id = symbol_info[0][0]
industry_id = symbol_info[0][1]

query = f"SELECT r.stock_symbol, l.price_datetime, l.open_price, l.high_price, l.low_price, l.close_price, l.volume, l.dividends, l.stock_splits FROM eod_price_history l INNER JOIN stock r ON r.stock_id = l.stock_id WHERE r.sector_id = {sector_id} OR r.industry_id = {industry_id};"

symbols = conn.execute('SELECT stock_symbol FROM stock')
symbols = symbols.fetchall()
symbols = [i[0] for i in symbols]
symbols = [i for i in symbols if i not in symbols]

df = pd.read_sql(query, conn, index_col=['stock_symbol', 'price_datetime'])

# next
# reset the index
df = df.reset_index()

# next
# convert to datetime
df['price_datetime'] = pd.to_datetime(df['price_datetime'], format='%Y-%m-%d')

# reshape
df = df.set_index(['price_datetime', 'stock_symbol']).unstack(['stock_symbol'])

df = df.loc[start_date:current_date]  # date range from 2019-01-01 to 2021-05-31

# set a threshold of how many np.nan you are willing to work with
close_df = df['close_price'].dropna(thresh=(len(df['close_price'] / 0.2)), axis=1)

# forward fill in np.nan values in columns not previously dropped.
close_df = close_df.fillna(method='ffill', axis=1)

# remove outliers
low_outlier = close_df.quantile(.1, axis=1).quantile(.1)
high_outlier = close_df.quantile(.9, axis=1).quantile(.9)
for column in close_df.columns:
    if (close_df[column].median() < low_outlier) or (close_df[column].median() > high_outlier):
        close_df = close_df.drop([column], axis=1)


# %%
# splitting test/training data
data_size = len(close_df)

# training_data = close_df.iloc[:(int(data_size * model_settings['train_test_ratio']))]
# test_data = close_df.iloc[(int(data_size * model_settings['train_test_ratio'])):]
training_data = close_df

# First, create a singular dataframe from the symbol you would like to predict vaues for
training_data_predicted = training_data[symbol]
training_data_predicted = training_data_predicted.to_frame()

minmax = MinMaxScaler(feature_range=(0,1))
minmax_single = MinMaxScaler(feature_range=(0,1))

# scale data
training_data = training_data.drop(columns=symbol)
train_scaled = np.concatenate([minmax.fit_transform(training_data.values), minmax_single.fit_transform(training_data_predicted.values)], axis=1)

columns = [i for i in training_data.columns]
columns.append(symbol)

# I am apparently missing all of the predicted symbol when later merging my data, maybe this will help?
training_data[symbol] = training_data_predicted[symbol]

train_scaled = pd.DataFrame(train_scaled, columns=columns)

train_scaled = train_scaled.values
n_lags = int(60) # not sure why hard 60
X_train = []
y_train = []
for i in range(n_lags, train_scaled.shape[0]):
    X_train.append(train_scaled[i-n_lags:i])
    y_train.append(train_scaled[i,0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], -1))


# %%
# TRAINING THE MODEL

# LSTM architecture
regressor = tf.keras.Sequential()

# First layer, w/ dropout regularization
regressor.add(tf.keras.layers.LSTM(units=model_settings['units'], return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
regressor.add(tf.keras.layers.Dropout(0.2))

# Second
regressor.add(tf.keras.layers.LSTM(units=model_settings['units'], return_sequences=True))
regressor.add(tf.keras.layers.Dropout(0.2))

# Third
regressor.add(tf.keras.layers.LSTM(units=model_settings['units'], return_sequences=True))
regressor.add(tf.keras.layers.Dropout(0.5))

# Fourth
regressor.add(tf.keras.layers.LSTM(units=model_settings['units']))
regressor.add(tf.keras.layers.Dropout(0.5))

# Output layer
# based on other tutorials, setting Dense units to # of days outward I would like to predict
regressor.add(tf.keras.layers.Dense(units=60))

# Compile LSTM
regressor.compile(optimizer='adam', loss='mean_squared_error')


# %%
print("training model...")
regressor.fit(X_train, y_train, epochs=model_settings['epochs'], batch_size=model_settings['batch_size'], verbose=1)

print("Saving Model...")
regressor.save(f"model_{current_datetime}")
print("Model Saved.")


# %%
# shape data to predict w/ 
'''# First, create a singular dataframe from the symbol you would like to predict vaues for
training_data_predicted = training_data[symbol]
training_data_predicted = training_data_predicted.to_frame()

minmax = MinMaxScaler(feature_range=(0,1))
minmax_single = MinMaxScaler(feature_range=(0,1))

# Replace FCEL w/ the symbol you'd like to predict, also apply this elsewhere
training_data = training_data.drop(columns=symbol)
train_scaled = np.concatenate([minmax.fit_transform(training_data.values), minmax_single.fit_transform(training_data_predicted.values)], axis=1)

columns = [i for i in training_data.columns]
columns.append(symbol)

# I am apparently missing all of the predicted symbol when later merging my data, maybe this will help?
training_data[symbol] = training_data_predicted[symbol]

train_scaled = pd.DataFrame(train_scaled, columns=columns)

# this section may not be necessary
train_scaled = train_scaled.values
n_lags = int(60) # not sure why hard 60
X_train = []
y_train = []
for i in range(n_lags, train_scaled.shape[0]):
    X_train.append(train_scaled[i-n_lags:i])
    y_train.append(train_scaled[i,0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], -1))

# this may also not be necessary'''

total_data = training_data
# inputs = total_data[len(total_data) - len(test_data) - 60:]
inputs = total_data[len(total_data) - 60:]
# inputs = data[len(data) - len(test_data) - 60:]

predicted_input = inputs[symbol].to_frame()
inputs = inputs.drop(columns=symbol)

test_scaled = np.concatenate([minmax.transform(inputs), minmax_single.transform(predicted_input)], axis=1)
print('Shape of test data:', test_scaled.shape)

# shaping test data from neural network
X_test = []
y_test = []
for i in range(60, test_scaled.shape[0]):
    X_test.append(test_scaled[i-n_lags:i])
    y_test.append(test_scaled[i,0])

X_test, y_test = np.array(X_test), np.array(y_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], -1))


# %%
# this will require some manually debugging, I'm sure of it

# this part is absolutely necessary
predicted_price = regressor.predict(X_test)
predicted_price = minmax_single.inverse_transform(predicted_price)
predicted_price = pd.DataFrame(predicted_price)
predicted_price.rename(columns = {0: f'{symbol}_predicted'}, inplace=True)
predicted_price = predicted_price.round(decimals=2)
predicted_price.index = test_data.index

# plot
plt.figure(figsize=(14,5))
#mse = mean_squared_error(y_test, predicted_price)
plt.plot(training_data[symbol], color = 'blue', label=f'{symbol} Price, training')
plt.plot(predicted_price[f'{symbol}_predicted'], color = 'red', label= f'Predicted {symbol} Closing Price')
plt.plot(test_data[symbol], color = 'green', label = f'Actual {symbol} Closing Price')
# plt.title("FCEL Closing PRice Prediction w/ MSE {:10:4f}".format(mse))
plt.title(f'{symbol} Closing Price Prediction')
plt.xlabel("Time")
plt.ylabel('Price USD')
plt.legend()
plt.savefig(f"{path}/{symbol}_{current_datetime}")

mse = mean_squared_error(test_data[symbol], predicted_price[f'{symbol}_predicted'])
mse_values[symbol] = mse

mse_values.to_csv(f"{path}/{current_datetime}_mse.csv")


