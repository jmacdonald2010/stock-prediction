# this script is used to extract price history data from yfinance
# the script is set to run until stopped
# while this won't be used in the final version of the application,
# it is acting as both a test to make sure that this code works, 
# and also as a way to gather data for when it is time to 
# start training the models.

from stock import Stock
from stock import Price
from stock import get_current_datetime
import sqlite3
from sqlite3 import Error
import yfinance as yf
import datetime
import time
import pandas as pd
import pickle
import numpy as np
from sklearn.impute import KNNImputer

# create an SQLite db connection
conn = sqlite3.connect('stockPrediction.db')

# collect stock symbols from DB, add to a list
dbq = conn.execute('SELECT stock_symbol FROM stock')
conn.commit()
dbq = dbq.fetchall()
# the result from dbq.fetchall() is a list of single-element tuples
# the list comprehension below turns it into a list of strings instead
db_symbols = [x[0] for x in dbq]

# create a dict of stock symbols and their stock_id
dbq = conn.execute('SELECT stock_id, stock_symbol FROM stock')
conn.commit()
dbq = dbq.fetchall()
stock_id_dict = dict()  # keys are ticker symbols, vals are the ID #
    
for x in dbq:
    stock_id_dict[x[1]] = x[0]

try:
    infile = open('gphl_symbol', 'rb')
    pickle_symbol = pickle.load(infile)
    pickle_symbol_read = False
except:
    pickle_symbol = None
    pickle_symbol_read = True

while True:

    for symbol in db_symbols:

        # first, check to see if we have a last symbol stored in a pickle    
        if pickle_symbol_read is False:
            if symbol == pickle_symbol:
                pickle_symbol_read = True
            else:
                continue

        # template for the DB query for writing new data to the db
        # price_history_query = "INSERT INTO price_history (stock_id, price_datetime, open_price, high_price, low_price, close_price, volume, dividends, stock_splits, datetime_added) VALUES "

        # collect price history data that already exists in the DB to avoid redundant data
        data_in_db = conn.execute(f'SELECT price_datetime FROM price_history WHERE stock_id = {stock_id_dict[symbol]}')
        conn.commit()
        data_in_db = data_in_db.fetchall()
        # compile a list of prev. price_datetime values
        prev_datetimes = [x[0] for x in data_in_db]

        # set the data range for the API call
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        sixty_days_ago = (datetime.datetime.now() - datetime.timedelta(days=59)).strftime("%Y-%m-%d")

        # API call
        data = yf.Ticker(symbol)
        data = data.history(
            start = sixty_days_ago,
            end = current_date,
            interval = '15m',
            auto_adjust = True
        )

        # add columns for datetime retrived and stock_ID
        # for this test, stock_id is static b/c the symbol we're using is static
        data['datetime_added'] = get_current_datetime()
        data['stock_id'] = stock_id_dict[symbol]

        # turns the datetime index to a column, so it's actually useful for us
        data.reset_index(level=0, inplace=True)

        # rename to match our db
        data = data.rename(columns={
            'Datetime': 'price_datetime',
            'Open': 'open_price', 
            'High': 'high_price', 
            'Low': 'low_price',
            'Close': 'close_price',
            'Volume': 'volume',
            'Dividends': 'dividends',
            'Stock Splits': 'stock_splits'
        })

        # convert datetime values to string
        data['price_datetime'] = data['price_datetime'].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))

        # fill in np.nan's
        impute_columns = []
        imputer = KNNImputer(n_neighbors=2, weights='uniform')
        imputed_data = imputer.fit_transform(data[['open_price', 'high_price', 'low_price', 'close_price', 'volume', 'dividends']])
        clean_data = pd.DataFrame(imputed_data, columns=['open_price', 'high_price', 'low_price', 'close_price', 'volume', 'dividends'])

        # copy the cleaned data into our dataframe
        data['open_price'] = clean_data['open_price']
        data['high_price'] = clean_data['high_price']
        data['low_price'] = clean_data['low_price']
        data['close_price'] = clean_data['close_price']
        data['volume'] = clean_data['volume']
        data['dividends'] = clean_data['dividends']

        # symbol = Stock(symbol)

        # variables to keep track of redundant data, new data when running this code
        new_data = 0
        redundant_data = 0

        # loop thru the rows, checking to see if we would have duplicate price_datetime values
        # if we do, then remove the row
        for row in data.itertuples():
            
            if prev_datetimes.count(row[1]) > 0:
                redundant_data += 1
                data = data.drop(row.Index)
                redundant_data += 1
            else:
                new_data += 1

        # pickle, for restarting the loop if it crashes
        outfile = open('gphl_symbol', 'wb')
        pickle.dump(symbol,outfile)
        outfile.close()

        # execute the query
        if new_data > 0:
            data.to_sql('price_history', conn, if_exists='append', index=False)
            conn.commit()

        print(symbol, " price history added to database.")
        print(new_data, " new entries added.")
        print(redundant_data, " redundant entries, not added.")
        print('Data added', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        # outfile.open()

        # delay to help us not get banned from yf
        time.sleep(5)