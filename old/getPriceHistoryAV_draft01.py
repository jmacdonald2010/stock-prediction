# imports
import mysql.connector
from mysql.connector import Error
#import pandas as pd
from getpass import getpass
import requests
# import json
import datetime
import time
import csv
import config

def create_server_connection(host_name, user_name, user_password):
    connection = None
    try:
        connection = mysql.connector.connect(host=host_name, user = user_name, passwd = user_password)
        print('mySQL database connection successful!')
    except Error as err:
        print(f"Error:'{err}'")
    return connection

def create_db_connection(host_name, user_name, user_password, db_name):
    connection = None
    try:
        connection = mysql.connector.connect(
            host=host_name,
            user=user_name,
            passwd=user_password,
            database=db_name
        )
        print('MySQL Connection Successful')
    except Error as err:
        print(f"Error: '{err}'")
        
    return connection

def execute_query(connection, query):
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        connection.commit()
        print('Query Successful')
    except Error as err:
        print(f"Error: '{err}'")

# test for reading db 

def read_query(connection, query):
    cursor = connection.cursor()
    result = None
    try:
        cursor.execute(query)
        result = cursor.fetchall()
        return result
    except Error as err:
        print(f"Error: '{err}'")

def change_brackets_to_parenthesis(data):
    return str(data).replace("[", "(").replace("]", ")")

def symbols_to_list(db_call):
    symbols = []
    for symbol in db_call:
        symbol_to_strip = str(symbol)
        symbol_to_append = symbol_to_strip.strip("(").strip(")").strip("\'").strip(",").strip("'")
        symbols.append(symbol_to_append)
    return symbols

fhand = open('priceHistory_symbolsAlreadyFilled.txt')
redundant_symbols = []
for line in fhand:
    redundant_symbol = line.strip("\n")
    redundant_symbols.append(redundant_symbol)
    print(redundant_symbol)

# main code below

# connect to db
connection = create_db_connection(config.db_ip, config.db_username, config.db_pw, config.db_name)

# read exising ticker symbols in db, compile into list
read_symbols_in_db = "SELECT tickerID FROM stockInfo;"
db_call = read_query(connection, read_symbols_in_db)
symbols = symbols_to_list(db_call)

# time periods to call, will likely result in excess unnecessary calls for missing data
month_year = ['year1month1', 'year1month2', 'year1month3', 'year1month4', 'year1month5', 'year1month6', 'year1month7', 'year1month8', 'year1month9', 'year1month10', 'year1month11', 'year1month12', 'year2month1', 'year2month2', 'year2month3', 'year2month4', 'year2month5', 'year2month6', 'year2month7', 'year2month8', 'year2month9', 'year2month10', 'year2month11', 'year2month12']

i = 0

for symbol in symbols:
    print(symbol)

    if symbol in redundant_symbols:
        continue

    for month in month_year:
        print("performing API call for ", symbol, "for", month)

        # api call
        call = {'symbol':symbol, 'interval':'15min', 'slice': month, 'apikey':config.av_api_key}
        resp = requests.get('https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&', params=call)

        i += 1

        print('made', i, 'api calls so far')

        # get list of all the datetime data we have for the selected stock
        # this is in order to prevent writing duplicate data to the db

        prev_read_query = "SELECT stockInfo_tickerID, priceDateTime FROM priceHistory WHERE stockInfo_tickerID = '" + str(symbol) + "' ;"
        prev_price_history = read_query(connection, prev_read_query)

        # create an empty list for the datetime data for the stock we're looping over
        prev_datetimes = []

        for entry in prev_price_history:
            prev_price_tuple = entry[1]
            #print(type(prev_price_tuple)) # just a test thing
            #prev_datetime = prev_price_tuple[1]
            prev_datetimes.append(prev_price_tuple.strftime("%Y-%m-%d %H:%M:%S"))

        data = resp.text.splitlines()
        reader = csv.reader(data)
        header = next(reader)
        if header!= None:
            for row in reader:
                if row[0] == '':
                    print("No Data")
                    break
                created_date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                price_datetime = row[0]
                openPrice = row[1]
                closePrice = row[4]
                highPrice = row[2]
                lowPrice = row[3]
                volume = row[5]
                
                if price_datetime in prev_datetimes:
                    print('redundant data, skipping')
                    continue
                
                price_history_info = symbol, created_date, price_datetime, openPrice, closePrice, highPrice, lowPrice, volume, 1
                
                pop_priceHistory = 'INSERT INTO priceHistory (stockInfo_tickerID, createdDate, priceDateTime, openPrice, closePrice, priceHigh, priceLow, volume, priceHistory_vendorID) VALUES ' +  str(price_history_info) + ';'
                
                execute_query(connection, pop_priceHistory)

                # i += 1

                if i > 475:
                    print('breaking due to # of api calls made')
                    break

        time.sleep(20)

        if i > 475: # note for 1/10/21, changing to 250 b/c of a lot of testing today, change back to 400 on any other day
            print('break due to # of api calls made')
            break

    if i > 475:
        print('break due to # of APi calls made')
        break


#print('last call was for', symbol, 'at', )
print('Wrote 400 new API calls of data to db')
        




