# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# Going to try making a univariate multi-step forecasting model,just train the model on all of the ticker symbols.
# 
# If this works well, I am going to delete all of the above cells, as I'm tired of making new notebooks to try different things.

# %%
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
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import random
import json

# load model settings for testing
settings = pd.read_csv('mlTests_settings - Sheet1.csv')
epochs = []
batch_sizes = []
n_steps_in_amts = []
# start_dates = settings['start_date']
n_steps_out = 60

for row in settings.itertuples():
    try:
        epochs.append(int(row[1]))
    except:
        pass
    try:
        batch_sizes.append(int(row[2]))
    except:
        pass
    try:
        n_steps_in_amts.append(int(row[4]))
    except:
        pass


# settings
# model_settings = {'epochs': 10, 'batch_size': 100, 'train_test_ratio': 0.7, 'hidden_layers': 3, 'units': 100, 'start_date': '2020-01-01', 'n_steps_in': 10, 'n_steps_out': 30, 'symbol': 'CTXR'}
current_date = datetime.datetime.now().strftime("%Y-%m-%d")

# load and shape data
conn = sqlite3.connect('stockPrediction_06072021.db')

symbol_info = conn.execute(f"SELECT sector_id, industry_id FROM stock WHERE stock_symbol = \"CTXR\";")
symbol_info = symbol_info.fetchall()
sector_id = symbol_info[0][0]
industry_id = symbol_info[0][1]

query = f"SELECT r.stock_symbol, l.price_datetime, l.open_price, l.high_price, l.low_price, l.close_price, l.volume, l.dividends, l.stock_splits FROM eod_price_history l INNER JOIN stock r ON r.stock_id = l.stock_id WHERE r.sector_id = {sector_id} OR r.industry_id = {industry_id};"

symbols = conn.execute('SELECT stock_symbol FROM stock')
symbols = symbols.fetchall()
symbols = [i[0] for i in symbols]
symbols = [i for i in symbols if i not in symbols]

df = pd.read_sql(query, conn, index_col=['stock_symbol', 'price_datetime'])
df = df.reset_index()

df['price_datetime'] = pd.to_datetime(df['price_datetime'], format='%Y-%m-%d')

df = df.set_index(['price_datetime', 'stock_symbol']).unstack(['stock_symbol'])

df = df.loc['2020-01-01':current_date]  # date range from 2019-01-01 to 2021-05-31

close_df = df['close_price'].dropna(thresh=(len(df['close_price'] / 0.2)), axis=1)

close_df = close_df.fillna(method='ffill', axis=1)

# remove outliers
low_outlier = close_df.quantile(.1, axis=1).quantile(.1)
high_outlier = close_df.quantile(.9, axis=1).quantile(.9)
for column in close_df.columns:
    if (close_df[column].median() < low_outlier) or (close_df[column].median() > high_outlier):
        close_df = close_df.drop([column], axis=1)
columns = [i for i in close_df.columns]
# close_df


# %%
# func to split the time sequence into samples
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


# %%
# normalize data
# here will go the function that will prepare the X, y data when fitting the model
# normalize, split_sequence, reshape
global scaler_dict
scaler_dict = {}
def prep_data(data, n_steps_in, n_steps_out):
    # data is a dataframe w/ one column
    df = data.to_frame()
    symbol = df.columns[0]
    scaler_dict[symbol] = MinMaxScaler(feature_range=(0,1))
    scaled = scaler_dict[symbol].fit_transform(df.values)
    X, y = split_sequence(scaled, n_steps_in, n_steps_out)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y


for i in epochs:
    for a in batch_sizes:
        for b in n_steps_in_amts:


            # %%
            # split train/test sets
            data_size = len(close_df)

            # using a 90/10 train/test split
            training_data = close_df[:(int(data_size * .7))]
            # this below might help me not run into issues reshaping data for predictions
            # might actually not be necessary now that I've figured out my predictions errors
            # while (len(training_data) % model_settings['n_steps_in']) != 0:
                # training_data = training_data.iloc[1:]
            test_data = close_df[(int(data_size * .7)):]

            # comment out the below line if you want a specific number of predictions, this is mainly useful to see test vs predicted data
            n_steps_out = len(test_data)


            # %%
            # define model
            model = Sequential()
            model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(b, 1)))
            model.add(LSTM(100, activation='relu'))
            model.add(Dense(n_steps_out))
            model.compile(optimizer='adam', loss='mse')


            # %%
            # fit model
            print(f'Starting train/test of model with epochs: {i}, batch size: {a}, n_steps_in: {b}')
            for epoch in range(i):
                print('Epoch #: ', epoch)
                for symbol in close_df.columns:
                    # print('Symbol: ', symbol)
                    x_train, y_train = prep_data(training_data[symbol], b, n_steps_out)
                    model.fit(x_train, y_train, batch_size=a, epochs=epoch+1, initial_epoch=epoch, verbose=0)


            # save model
            print('Training complete. Saving model...')

            current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            model.save(f"model_{current_datetime}")
            print('Model saved.')
            print('Beginning tests...')

            # predict a symbol
            # we will be predicting five symbols and comparing/contrasting the results of those
            test_symbols = ["CTXR", "IBIO", "MNKD", "IMGN", "ATOS", "SPPI"]

            model_mse_values = {}
            model_mse_values['model_values'] = {
                'epochs': i,
                'batch_size': a,
                'n_steps_in': b,
                'n_steps_out': n_steps_out
            }

            print('Creating a new directory to save plots to...')
            # create a folder to save this all in
            directory = f"test_plots_{current_datetime}"
            # change the parent_dir when running on different machines
            # windows path
            parent_dir = r"C:\Users\james\Dropbox\Box Sync\code\stockPrediction"
            # mac path
            # parent_dir = r"/Users/jamesmacdonald/Dropbox/Box Sync/code/stockPrediction"
            path = os.path.join(parent_dir, directory)
            os.mkdir(path)
            print('Directory made.')
            print('Testing symbols...')


            # %%
            # script that predicts five symbols at random

            for s in test_symbols:
                print('Testing symbol', s)
                
                # shape of the single sample of input data when making the prediction must be 1 sample, the # of input steps, and the single feature.
                # symbol = s
                x_input = training_data[s].to_frame()
                x_input = scaler_dict[s].fit_transform(x_input.values)
                x_input = x_input[-b:]
                # next line, b/c the data they feed there's before reshaping is 1D
                x_input = x_input.reshape((-1))
                x_input = x_input.reshape((1, b, 1))
                yhat = model.predict(x_input)
                yhat = scaler_dict[symbol].inverse_transform(yhat)
                print(yhat)


                # %%
                # prep a dataframe to make all of the data compatible
                yhat = yhat.reshape((-1))
                predicted = pd.DataFrame(yhat, columns=[s])
                test_data = test_data.reset_index()
                '''while len(predicted) > len(test_data):
                    last_date = test_data['price_datetime'].iloc[-1]
                    future_date = last_date + datetime.timedelta(days=1)
                    while future_date.weekday() in [5,6]:
                        future_date = future_date + datetime.timedelta(days=1)
                    future_date = future_date.strftime('%Y-%m-%d')
                    future_date = pd.to_datetime(future_date, format='%Y-%m-%d')
                    future_dates = {'price_datetime': future_date}
                    test_data = test_data.append(future_dates, ignore_index=True)'''

                predicted['price_datetime'] = test_data['price_datetime']
                test_data = test_data.set_index('price_datetime')
                predicted = predicted.set_index('price_datetime')

                # create the plot
                plt.figure(figsize=(14,5))
                plt.plot(training_data[s], color='blue', label=f"{s} price, training data")
                plt.plot(test_data[s], color='red', label=f"{s} price, test data")
                plt.plot(predicted[f"{s}"], color='green', label=f"{s} price, predicted data")
                plt.legend()
                plt.savefig(f"{path}/{s}_{current_datetime}")
                mse = mean_squared_error(test_data[s], predicted[s])
                model_mse_values[s] = mse

            # write the model_mse_values dict to a file for reference
            with open(f'{path}/model_mse_values.txt', 'w') as file:
                file.write(json.dumps(model_mse_values))
            print('Saved model_mse_values to text file.')
            print(f'Completed train/test of model with epochs: {i}, batch size: {a}, n_steps_in: {b}')

print('Trained all given options.')


