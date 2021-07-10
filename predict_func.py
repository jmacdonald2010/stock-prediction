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
    
    def __init__(self, mse=.002):
        super().__init__()
        self.mse = mse

    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('mse') < self.mse):
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

def predict_symbol(symbol, epochs=2000, batch_size=100, start_date='2020-01-01', n_steps_in=5, n_steps_out=30, training_mse=.002):

    # settings
    # good universal set of settings, at least for RSSS
    #model_settings = {'epochs': 250, 'batch_size': 100, 'train_test_ratio': 0.7, 'hidden_layers': 3, 'units': 200, 'start_date': '2020-01-01', 'n_steps_in': 90, 'n_steps_out': 60, 'symbol': 'DTGI'}
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")

    # settings to mess w/
    # uncomment, comment as needed
    # 5 steps in is best for EOD, 30 steps out
    model_settings = {'epochs': epochs, 'batch_size': batch_size, 'train_test_ratio': 0.7, 'hidden_layers': 3, 'units': 200, 'start_date': start_date, 'n_steps_in': n_steps_in, 'n_steps_out': n_steps_out, 'symbol': symbol, 'interval': '1d'}

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
    close_df = df['Close'].to_frame()

    print(close_df.isna().sum())

    # split train/test sets
    data_size = len(close_df)

    # using a 90/10 train/test split
    training_data = close_df[:(int(data_size * .7))]
    test_data = close_df[(int(data_size * .7)):]

    # here is where I'm going to scale the test data
    scaler = MinMaxScaler(feature_range=(0,1))

    training_data_scaled = scaler.fit_transform(training_data.to_numpy())

    training_data_scaled = pd.DataFrame(training_data_scaled)

    test_data_scaled = scaler.transform(test_data.to_numpy())
    test_data_scaled = pd.DataFrame(test_data_scaled)
    print(test_data_scaled.describe())

    # comment out the below line if you want a specific number of predictions, this is mainly useful to see test vs predicted data
    model_settings['n_steps_out'] = len(test_data)


    # define model
    model = Sequential()
    model.add(LSTM(200, activation='relu', return_sequences=True, input_shape=(model_settings['n_steps_in'], 1)))
    model.add(LSTM(200, activation='relu'))
    model.add(Dense(model_settings['n_steps_out']))
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    callbacks = [new_callback(mse=training_mse), EarlyStopping(monitor='loss', patience=45)]

    # fit model

    x_train, y_train = prep_data(training_data_scaled, model_settings['n_steps_in'], model_settings['n_steps_out'])
    model.fit(x_train, y_train, batch_size=model_settings['batch_size'], epochs=model_settings['epochs'], verbose=1, shuffle=True, callbacks=callbacks)

    # predict for train/test purposes
    symbol = model_settings['symbol']
    # x_input = training_data[symbol].to_frame()
    # x_input = scaler_dict[symbol].fit_transform(x_input.values)
    # x_input, single_scaler = prep_single_symbol(symbol, training_data)
    x_input = training_data_scaled.to_numpy()
    x_input = x_input[-model_settings['n_steps_in']:]
    # next line, b/c the data they feed there's before reshaping is 1D
    x_input = x_input.reshape((-1))
    x_input = x_input.reshape((1, model_settings['n_steps_in'], 1))
    yhat = model.predict(x_input)
    # yhat = scaler_dict[symbol].inverse_transform(yhat)
    yhat = scaler.inverse_transform(yhat)
    print(yhat)

    yhat = yhat.reshape((-1))
    predicted = pd.DataFrame(yhat, columns=[symbol])
    test_data = test_data.reset_index()

    predicted['price_datetime'] = test_data['Date']
    test_data = test_data.set_index('Date')
    predicted = predicted.set_index('price_datetime')

    last_value = training_data['Close'].iloc[-1]
    difference = last_value - predicted[symbol].iloc[0]
    predicted = predicted + difference

    mse = mean_squared_error(test_data.to_numpy(), predicted.to_numpy())
    print('MSE:', mse)

    plt.figure(figsize=(14,5))
    plt.plot(training_data['Close'], color='blue', label=f"{symbol} price, training data")
    plt.plot(test_data['Close'], color='red', label=f"{symbol} price, test data")
    plt.plot(predicted[f"{symbol}"], color='green', label=f"{symbol} price, predicted data")
    plt.legend()
    plt.show()

    # ask if the user wants to predict using this or try again
    input_val = False

    while input_val == False:
        user_resp = input('Would you like to predict based on this model? Y/N ')
        if user_resp == 'Y':
            input_val = True
        elif user_resp == 'N':
            exit()
        else:
            print('Please input a valid response (Y/N)')

    # predict on all values
    symbol = model_settings['symbol']
    x_input = scaler.transform(close_df.to_numpy())
    x_input = x_input[-model_settings['n_steps_in']:]
    x_input = x_input.reshape((-1))
    x_input = x_input.reshape((1, model_settings['n_steps_in'], 1))
    yhat = model.predict(x_input)
    yhat = scaler.inverse_transform(yhat)
    print(yhat)

    # add dates to predicted dataframe
    # only use if not testing
    # only works w/ EOD values
    yhat = yhat.reshape((-1))
    predicted = pd.DataFrame(yhat, columns=[symbol])
    # training_data = training_data.reset_index()
    close_df = close_df.reset_index()
    for i in range(len(predicted)):
        try:
            future_date = predicted['Date'].iloc[i -1] + datetime.timedelta(days=1)
        except:
            future_date = close_df.Date.iloc[-1] + datetime.timedelta(days=1)
        
        while future_date.weekday() in [5,6]:
            future_date = future_date + datetime.timedelta(days=1)
        future_date = future_date.strftime('%Y-%m-%d')
        future_date = pd.to_datetime(future_date, format='%Y-%m-%d')
        future_dates = {'Date': future_date}
        '''try:
            future_dates_df = future_dates_df.append(future_dates, ignore_index=True)
        except:
            future_dates_df = pd.DataFrame.from_dict(future_dates)'''
        predicted = predicted.append(future_dates, ignore_index=True)
        predicted['Date'].iloc[i] = future_date
    # predicted_price['price_datetime'] = future_dates_df['price_datetime']
    # this is so that when looping this code it doesn't cause problems
    # training_data = training_data.set_index('Date')
    close_df = close_df.set_index('Date')

    predicted = predicted.set_index('Date')
    predicted = predicted.dropna()

    # ONLY FOR NON-TESTING
    last_value = close_df['Close'].iloc[-1]
    difference = last_value - predicted[symbol].iloc[0]
    predicted = predicted + difference

    max_increase = ((predicted[symbol].max() - df['Close'].iloc[-1]) / df['Close'].iloc[-1] * 100)


    # plot w/o test data
    plt.figure(figsize=(14,5))
    plt.plot(close_df['Close'].iloc[-120:], color='blue', label=f"{symbol} price, training data")
    # plt.plot(test_data['Close'], color='red', label=f"{symbol} price, test data")
    plt.plot(predicted[f"{symbol}"].iloc[:60], color='green', label=f"{symbol} price, predicted data")
    plt.title(f'{symbol}_mse_{mse}_max_{predicted[symbol].max()}_maxDate_{predicted[symbol].idxmax()}_percent_to_max_{max_increase}')

    plt.legend()
    plt.show()

    print('Max:', predicted[symbol].max(), "Max Date:", predicted[symbol].idxmax(), 'Percent increase to max from current:', ((predicted[symbol].max() - df['Close'].iloc[-1]) / df['Close'].iloc[-1] * 100))


predict_symbol('LOVFF', n_steps_in=10, training_mse=.0002)