from database_functions import create_db_connection
from database_functions import execute_query
from database_functions import read_query
import yfinance as yf
import requests
import datetime
import time
from getpass import getpass
import config
# import numpy

def symbols_to_list(db_call):
    symbols = []
    for symbol in db_call:
        symbol_to_strip = str(symbol)
        symbol_to_append = symbol_to_strip.strip("(").strip(")").strip("\'").strip(",").strip("'")
        symbols.append(symbol_to_append)
    return symbols

# connect to the db
connection = create_db_connection(config.db_ip, config.db_username, config.db_pw, config.db_name)

# read symbols in the DB, append to a list
read_symbols_in_db = 'SELECT tickerID FROM stockInfo;'
db_call = read_query(connection, read_symbols_in_db)
symbols = symbols_to_list(db_call)

# start template for the INSERT INTO SQL cmd


for symbol in symbols:
    price_history_query = "INSERT INTO priceHistory (stockInfo_tickerID, createdDate, priceDateTime, openPrice, closePrice, priceHigh, priceLow, volume, priceHistory_vendorID) VALUES "
    
    # read datatime info from db
    data_in_db = read_query(connection, f"SELECT stockInfo_tickerID, priceDateTime FROM priceHistory WHERE stockInfo_tickerID = '{symbol}';")
    
    # create a list of prev_datetimes for the iterated symbol
    prev_datetimes = []
    for value in data_in_db:
        prev_datetimes.append(value[1].strftime("%Y-%m-%d %H:%M:%S"))

    # set the dates to get price history data from, then perform the API call
    current_date = datetime.datetime.now()
    current_date = current_date.strftime("%Y-%m-%d")
    sixty_days_ago = datetime.datetime.now() - datetime.timedelta(days = 59)
    sixty_days_ago = sixty_days_ago.strftime("%Y-%m-%d")
    data = yf.download(
        tickers = symbol,
        start = sixty_days_ago,
        end = current_date,
        interval = "15m"
    )

    # iterate thru each row in the price_history dataframe
    for row in data.itertuples():
        price_datetime = row[0]
        open_price = row[1]
        high_price = row[2]
        low_price = row[3]
        close_price = row[4]
        volume = row[6]

        price_datetime = price_datetime.strftime("%Y-%m-%d %H:%M:%S")

        # if a prev date_time is already in the db, just skip it to avoid redudancy
        # this statement works, turns out my data is a bit off in the DB 
        if prev_datetimes.count(price_datetime) > 0:
            print('redundant data, skipping')
            continue
        
        print("new data, adding to db query")

        # generate a datetime to put in for the entry
        created_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # add to the query string all of the data over reach
        price_history_query = price_history_query + f"\n('{symbol}', '{created_date}', '{price_datetime}', '{open_price}', '{close_price}', '{high_price}', '{low_price}', '{volume}', '2'),"
    
    # fix the last line of the query is remove the last comma, change it to a ;
    price_history_query = price_history_query[:-1]
    price_history_query = price_history_query + ";"
    # write the query to the DB
    execute_query(connection, price_history_query)
    time.sleep(5)
