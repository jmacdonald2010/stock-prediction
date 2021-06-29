# this is an early version of this file
# this will eventually be a full function, possibly in a kivy app

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import datetime
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import metrics
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

class new_callback(Callback):
    # courtesy of stackoverflow user Suvo, https://stackoverflow.com/questions/37293642/how-to-tell-keras-stop-training-based-on-loss-value
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('mse') < .002):
            print("90% Accuracy, stop training!")
            self.model.stop_training = True
            return

def split_sequence(sequence, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find end of pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check to see if we're beyond the sequence
        if out_end_ix > len(sequence):
            break
        # gather input, output parts of pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def prep_data(data, n_steps_in, n_steps_out):
    # data is a dataframe w/ one column
    # df = data.to_frame()
    # symbol = df.columns[0]
    # scaler_dict[symbol] = MinMaxScaler(feature_range=(0,1))
    # scaled = scaler_dict[symbol].fit_transform(df.values)
    X, y = split_sequence(data.to_numpy(), n_steps_in, n_steps_out)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y


current_date = datetime.datetime.now().strftime("%Y-%m-%d")

# settings to mess w/
# uncomment, comment as needed
# 5 steps in is best for EOD, 30 steps out
model_settings = {'epochs': 2000, 'batch_size': 100, 'train_test_ratio': 0.7, 'hidden_layers': 3, 'units': 200, 'start_date': '2020-01-01', 'n_steps_in': 30, 'n_steps_out': 30, 'symbol': 'SID', 'start_date': '2020-01-01', 'interval': '1d'}

# can't remember all intraday time intervals for the model settings
if model_settings['interval'] in ['1m', '5m', '10m', '15m', '30m', '1hr']:
    model_settings['start_date'] = (datetime.datetime.now() - datetime.timedelta(days=59)).strftime("%Y-%m-%d")

df = yf.Ticker(model_settings['symbol'])
df = df.history(
    start= model_settings['start_date'],
    end = current_date,
    interval= model_settings['interval'],
    auto_adjust= True
)

# close_df = df['close_price'].dropna(thresh=(len(df['close_price'] / 0.2)), axis=1)

close_df = df['Close'].to_frame()

print(close_df.isna().sum())

# close_df = close_df.fillna(method='ffill', axis=1)

# remove outliers
close_df


# split train/test sets
# run if testing model settings
data_size = len(close_df)

# using a 90/10 train/test split
training_data = close_df[:(int(data_size * .7))]
test_data = close_df[(int(data_size * .7)):]

# here is where I'm going to scale the test data
scaler = MinMaxScaler(feature_range=(0,1))
# scaler_single = MinMaxScaler(feature_range=(0,1))

# single_df = training_data[model_settings['symbol']].to_frame()

training_data_scaled = scaler.fit_transform(training_data.to_numpy())
# single_data = scaler_single.fit_transform(single_df.to_numpy())

training_data_scaled = pd.DataFrame(training_data_scaled)

test_data_scaled = scaler.transform(test_data.to_numpy())
test_data_scaled = pd.DataFrame(test_data_scaled)
print(test_data_scaled.describe())

# comment out the below line if you want a specific number of predictions, this is mainly useful to see test vs predicted data
model_settings['n_steps_out'] = len(test_data)

trainings_completed = 0
upper_test_completed = False
lower_test_completed = False
best_mse_found = False
n_steps_in_options = [2, 3, 4, 5, 10, 15, 30, 60, 90, 120]
n_steps_in_tested = []
mse_info = {}

while best_mse_found is False:

    # start by defining a model
    model = Sequential()
    model.add(LSTM(200, activation='relu', return_sequences=True, input_shape=(model_settings['n_steps_in'], 1)))
    model.add(LSTM(200, activation='relu'))
    model.add(Dense(model_settings['n_steps_out']))
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    callbacks = [new_callback(), EarlyStopping(monitor='loss', patience=45)]

    # now train the model
    x_train, y_train = prep_data(training_data_scaled, model_settings['n_steps_in'], model_settings['n_steps_out'])
    model.fit(x_train, y_train, batch_size=model_settings['batch_size'], epochs=model_settings['epochs'], verbose=1, shuffle=True, callbacks=callbacks)

    # run a prediction and calculate the MSE
    symbol = model_settings['symbol']
    x_input = training_data_scaled.to_numpy()
    x_input = x_input[-model_settings['n_steps_in']:]
    x_input = x_input.reshape((-1))
    x_input = x_input.reshape((1, model_settings['n_steps_in'], 1))
    yhat = model.predict(x_input)
    yhat = scaler.inverse_transform(yhat)
    print(f'Prediction for {symbol}', yhat)

    # make the dataframes have dates
    yhat = yhat.reshape((-1))
    predicted = pd.DataFrame(yhat, columns=[symbol])
    test_data = test_data.reset_index()

    predicted['price_datetime'] = test_data['Date']
    test_data = test_data.set_index('Date')
    predicted = predicted.set_index('price_datetime')

    # shift data to approx where it should start
    last_value = training_data['Close'].iloc[-1]
    difference = last_value - predicted[symbol].iloc[0]
    predicted = predicted + difference

    mse = mean_squared_error(test_data.to_numpy(), predicted.to_numpy())
    print('MSE for n_steps_in', model_settings['n_steps_in'], ':', mse)

    # this part will be a bit more complicated to write
    try:
        if mse < best_mse:
            best_mse = mse
    except:
        best_mse = mse
        
    trainings_completed += 1

    # document what we've tried so far
    # the dict may not be useful
    mse_info['n_steps_in_mse'] = mse, model_settings['n_steps_in']
    n_steps_in_tested.append(model_settings['n_steps_in''])

    if upper_test_completed is False:
        mse_pos = n_steps_in_options.index(model_settings['n_steps_in'])
        mse_pos += 1
        model_settings['n_steps_in'] = n_steps_in_options[mse_pos]
        upper_test_completed = True
    elif lower_test_completed is False:
        mse_pos = n_steps_in_options.index(model_settings['n_steps_in''])
        mse_pos -= 1
        model_settings['n_steps_in'] = n_steps_in_options[mse_pos]
        lower_test_completed = True
    else:
        if mse < best_mse:
            # determine if we have tested the next n_steps_in val up or down
            last_nsi = n_steps_in_tested[-1]
            if last_nsi > model_settings['n_steps_in']:
                # if we got a better value on a higher # of steps in, we want to test the next highest amt of steps in to see if there's additional improvement
                # first, compare and contrast the last mse val for steps in w/ the current
                pass # temp
                

