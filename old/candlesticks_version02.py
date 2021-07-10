import yfinance as yf
import matplotlib.pyplot as plt
import mplfinance as mpf
# from mplfinance import candlestick_ohlc
import pandas as pd
import matplotlib.dates as mpdates
import datetime
from database_functions import create_db_connection
from database_functions import execute_query
from database_functions import read_query
import config

symbol = 'MSFT' # for testing purposes


connection = create_db_connection(config.db_ip, config.db_username, config.db_pw, config.db_name)

symbol = input("Enter Symbol to lookup: ")
symbol = symbol.upper()
db_query = f"SELECT stockInfo_tickerID, priceDateTime, openPrice, priceHigh, priceLow, closePrice FROM priceHistory WHERE stockInfo_tickerID = '{symbol}';"

db_call = read_query(connection, db_query) # returns a list of tuples

# working on the db call first
'''
mpf.plot(data, type='candlestick', no_xgaps = True)

plt.show()'''