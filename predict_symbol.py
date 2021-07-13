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
import pickle

class new_callback(Callback):
    # courtesy of stackoverflow user Suvo, https://stackoverflow.com/questions/37293642/how-to-tell-keras-stop-training-based-on-loss-value
    def __init__(self, mse=.002):
        super().__init__()
        self.mse = mse

    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('mse') < self.mse):
            print("Stopping training, MSE threshold met.")
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
    X, y = split_sequence(data.to_numpy(), n_steps_in, n_steps_out)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y

def predict_symbol(symbol, epochs=2000, batch_size=100, start_date='2020-01-01', n_steps_in=5, n_steps_out=None, training_mse=.002, min_mse=3.5, min_percent_increase=0., show_downward_predictions=False, save_prediction_as_csv=False, is_watchlist_symbol=False):

    current_date = datetime.datetime.now().strftime("%Y-%m-%d")

    path = f'{current_date}_predictions'

    if not os.path.exists(path):
        os.makedirs(path)

    # create folder inside of that day's folder for watchlist symbols
    if is_watchlist_symbol:
        if not os.path.exists(f'{path}/watchlist_symbols'):
            os.makedirs(f'{path}/watchlist_symbols')

    model_settings = {'epochs': epochs, 'batch_size': batch_size, 'train_test_ratio': 0.7, 'hidden_layers': 3, 'units': 200, 'start_date': start_date, 'n_steps_in': n_steps_in, 'n_steps_out': n_steps_out, 'symbol': symbol, 'interval': '1d'}

    # can't remember all intraday time intervals for the model settings
    if model_settings['interval'] in ['1m', '5m', '10m', '15m', '30m', '1hr']:
        model_settings['start_date'] = (datetime.datetime.now() - datetime.timedelta(days=59)).strftime("%Y-%m-%d")

    try:
        df = yf.Ticker(model_settings['symbol'])
        df = df.history(
            start= model_settings['start_date'],
            end = current_date,
            interval= model_settings['interval'],
            auto_adjust= True
        )
    except:
        print(f'Error fetching data from yfinance for {symbol}.')
        return
    close_df = df['Close'].to_frame()

    if len(close_df) <= 0:
        print("No data for", symbol, "continuing to next.")
        return

    print(close_df.isna().sum())

    # split train/test sets
    data_size = len(close_df)

    # using a 90/10 train/test split
    training_data = close_df[:(int(data_size * .7))]
    test_data = close_df[(int(data_size * .7)):]

    # here is where I'm going to scale the test data
    scaler = MinMaxScaler(feature_range=(0,1))

    try:
        training_data_scaled = scaler.fit_transform(training_data.to_numpy())
    except:
        print('MinMaxScaler Error.')
        return

    training_data_scaled = pd.DataFrame(training_data_scaled)

    test_data_scaled = scaler.transform(test_data.to_numpy())
    test_data_scaled = pd.DataFrame(test_data_scaled)
    print(test_data_scaled.describe())

    # comment out the below line if you want a specific number of predictions, this is mainly useful to see test vs predicted data
    if model_settings['n_steps_out'] is None:
        model_settings['n_steps_out'] = len(test_data)


    # define model
    model = Sequential()
    model.add(LSTM(200, activation='relu', return_sequences=True, input_shape=(model_settings['n_steps_in'], 1)))
    model.add(LSTM(200, activation='relu'))
    model.add(Dense(model_settings['n_steps_out']))
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    callbacks = [new_callback(mse=training_mse), EarlyStopping(monitor='loss', patience=45)]

    # fit model

    print('Training symbol:', symbol)
    try:
        x_train, y_train = prep_data(training_data_scaled, model_settings['n_steps_in'], model_settings['n_steps_out'])
    except:
        print('Data shaping error, please inspect your data.')
        return
    try:
        model.fit(x_train, y_train, batch_size=model_settings['batch_size'], epochs=model_settings['epochs'], verbose=0, shuffle=True, callbacks=callbacks)
    except:
        print('Training error, please try again or inspect your data to check for errors.')
        return

    # predict for train/test purposes
    symbol = model_settings['symbol']
    x_input = training_data_scaled.to_numpy()
    x_input = x_input[-model_settings['n_steps_in']:]
    # next line, b/c the data they feed there's before reshaping is 1D
    x_input = x_input.reshape((-1))
    x_input = x_input.reshape((1, model_settings['n_steps_in'], 1))
    yhat = model.predict(x_input)
    
    # calculate the MSE based on the scaled values, not actual values
    # not entirely sure this is the best way to do this, but we'll see.
    yhat_test = yhat.reshape((-1))
    yhat_test = pd.DataFrame(yhat_test, columns=[symbol])
    try:
        if len(test_data_scaled) > len(yhat_test):
            mse = mean_squared_error(test_data_scaled[:len(yhat_test)].to_numpy(), yhat_test.to_numpy())
        elif len(test_data_scaled) < len(yhat_test):
            mse = mean_squared_error(test_data_scaled.to_numpy(), yhat_test[:len(test_data_scaled)].to_numpy())
        else:
            mse = mean_squared_error(test_data_scaled.to_numpy(), yhat_test.to_numpy())
    except:
        print("Error in MSE Calculation.")
        return
    if mse > min_mse:
        print('MSE too great, not making future prediction.')
        return
    yhat = scaler.inverse_transform(yhat)
    print(yhat)

    yhat = yhat.reshape((-1))
    predicted = pd.DataFrame(yhat, columns=[symbol])
    print(yhat.min())
    if yhat.min() < 0.:
        print('Prediction inaccurate; negative values in prediction. Please adjust training settings and try again.')
        return
    test_data = test_data.reset_index()

    predicted['price_datetime'] = test_data['Date']
    test_data = test_data.set_index('Date')
    predicted = predicted.set_index('price_datetime')

    last_value = training_data['Close'].iloc[-1]
    difference = last_value - predicted[symbol].iloc[0]
    predicted = predicted + difference

    # below commented out due to mse calculation being done on scaled values above
    # if successful, will be removed in a future version
    '''
    try:
        mse = mean_squared_error(test_data.to_numpy(), predicted.to_numpy())
    except:
        print('Error in MSE calculation.')
        return
    print('MSE:', mse)
    if mse > min_mse:
        print('MSE too great, not making future prediction.')
        return'''

    plt.figure(figsize=(14,5))
    plt.plot(training_data['Close'], color='blue', label=f"{symbol} price, training data")
    if len(test_data) > len(predicted):
        plt.plot(test_data['Close'][:len(predicted)], color='red', label=f"{symbol} price, test data")
        plt.plot(predicted[f"{symbol}"], color='green', label=f"{symbol} price, predicted data")
    elif len(test_data) < len(predicted):
        plt.plot(test_data['Close'], color='red', label=f"{symbol} price, test data")
        plt.plot(predicted[f"{symbol}"][:len(test_data)], color='green', label=f"{symbol} price, predicted data")
    else:
        plt.plot(test_data['Close'], color='red', label=f"{symbol} price, test data")
        plt.plot(predicted[f"{symbol}"], color='green', label=f"{symbol} price, predicted data")
    # plt.plot(test_data['Close'], color='red', label=f"{symbol} price, test data")
    # plt.plot(predicted[f"{symbol}"], color='green', label=f"{symbol} price, predicted data")
    plt.title(f'{symbol}_mse_{mse}_train_test')
    plt.legend()
    if is_watchlist_symbol:
        plt.savefig(f'{path}/watchlist_symbols/{symbol}_{current_date}_mse_{mse}_test.png')
    else:
        plt.savefig(f'{path}/{symbol}_{current_date}_mse_{mse}_test.png')

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
    # only works w/ EOD values
    yhat = yhat.reshape((-1))
    if yhat.min() < 0.:
        print('Prediction inaccurate; negative values in prediction. Please adjust training settings and try again.')
        return
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
        predicted = predicted.append(future_dates, ignore_index=True)
        predicted['Date'].iloc[i] = future_date
    close_df = close_df.set_index('Date')

    predicted = predicted.set_index('Date')
    predicted = predicted.dropna()

    # ONLY FOR NON-TESTING
    last_value = close_df['Close'].iloc[-1]
    difference = last_value - predicted[symbol].iloc[0]
    predicted = predicted + difference

    max_increase = ((predicted[symbol].max() - df['Close'].iloc[-1]) / df['Close'].iloc[-1] * 100)
    lowest_percent_decrease = ((predicted[symbol].min() - df['Close'].iloc[-1]) / df['Close'].iloc[-1] * 100)

    # determine if the minimum percent increase is enough to show and save the predicted plot
    if max_increase < min_percent_increase:
        if show_downward_predictions is False:
            print('Symbol not projected to increase in price during the predicted time frame. Please enter a different symbol or adjust the prediction settings.')
            return

    # save prediction as csv if true
    if save_prediction_as_csv is True:
        if is_watchlist_symbol:
            predicted.to_csv(f'{path}/watchlist_symbols/{symbol}_prediction_{current_date}')
        else:
            predicted.to_csv(f'{path}/{symbol}_prediction_{current_date}')
        print('Saved prediction to csv')

    # plot w/o test data
    plt.figure(figsize=(14,5))
    # this setting below could probably be adjusted
    plt.plot(close_df['Close'].iloc[-120:], color='blue', label=f"{symbol} price, training data")
    # plt.plot(test_data['Close'], color='red', label=f"{symbol} price, test data")
    plt.plot(predicted[f"{symbol}"], color='green', label=f"{symbol} price, predicted data")
    plt.title(f'{symbol}_mse_{mse}_max_{predicted[symbol].max()}_maxDate_{predicted[symbol].idxmax()}_percent_to_max_{max_increase}_minDate_{predicted[symbol].idxmin()}_percent_to_min_{lowest_percent_decrease}', fontsize=8)
    plt.legend()
    if is_watchlist_symbol:
        plt.savefig(f'{path}/watchlist_symbols/{symbol}_{current_date}_mse_{mse}_real_prediction.png')
    else:
        plt.savefig(f'{path}/{symbol}_{current_date}_mse_{mse}_real_prediction.png')

    # create stats dict
    stats = {
        'mse': mse,
        'max_percent_increase': max_increase,
        'max_increase_date': predicted[symbol].idxmax(),
        'max_percent_decrease': lowest_percent_decrease,
        'max_decrease_date': predicted[symbol].idxmin()
    }

    print('Completed train/test and future prediction for', symbol)
    return stats


# this is to test the function
print(predict_symbol('LPCN', show_downward_predictions=True, save_prediction_as_csv=True, is_watchlist_symbol=False))