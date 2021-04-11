from stock import Stock
from stock import Price
from stock import get_current_datetime
import sqlite3
from sqlite3 import Error
import yfinance as yf
import datetime
import time
import pandas as pd

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

for symbol in db_symbols:
    
    # template for the DB query for writing new data to the db
    price_history_query = "INSERT INTO price_history (stock_id, price_datetime, open_price, high_price, low_price, close_price, volume, dividends, stock_splits, datetime_added) VALUES "

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

    symbol = Stock(symbol)

    # variables to keep track of redundant data, new data when running this code
    new_data = 0
    redundant_data = 0

    for row in data.itertuples():
        
        # create a price object w/ the necessary arguments
        price_data = Price(symbol, row[1], row[2], row[3], row[4], row[5], row[0], row[6], row[7])
        # print(price_data)
        # print("")

        # convert the price_datetime 
        price_data.price_datetime = price_data.price_datetime.strftime("%Y-%m-%d %H:%M:%S")

        # if a prev_datetime is already in the DB, skip the iteration
        if prev_datetimes.count(price_data.price_datetime) > 0:
            redundant_data += 1
            continue

        # this part of the loop will only run for new data
        new_data += 1

        # entry to get current datatime to add to the DB query
        datetime_added = get_current_datetime()

        # add the data to the query string
        price_history_query = price_history_query + f"\n({stock_id_dict[symbol.symbol]}, '{price.price_datetime}', {price.open_price}, {price.high_price}, {price.low_price}, {price.close_price}, {price.volume}, {price.dividends}, '{price.stock_splits}', '{datetime_added}' ),"

    # after the itertuples loop
    # target the last line of the DB query, 
    price_history_query = price_history_query[:-1]
    price_history_query = price_history_query + ";"

    # execute the query
    conn.execute(price_history_query)
    conn.commit()

    # delay to help us not get banned from yf
    time.sleep(5)