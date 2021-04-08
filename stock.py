# creating a stock class to make things more organized?

import sqlite3
import datetime
import time

class Stock:

    def __init__(self, symbol, short_name=None, full_name=None, sector=None, industry=None, sector_id=None, industry_id=None):
        self.symbol = symbol
        # Everything else below here does not need to be defined upon creation
        self.short_name = short_name
        self.full_name = full_name
        self.sector = sector
        self.industry = industry
        self.sector_id = sector_id
        self.industry_id = industry_id

    def check_industry(self, industry_name, conn):
        '''This method takes the industry given as an input and checks to see if it exists in the database. If it does not already exist in the database, it adds it to the database.'''

        industries = conn.execute('SELECT industry FROM industry')
        industries = industries.fetchall()
        
        # loop thru all of the different industries in the industry table
        # if the industry is found, we exit the while loop w/o adding it.
        # if the industry is not found, we add it to the db
        x = False
        for i in industries:
            if i == industry_name:
                x = True
                break
        if x is False:
            conn.execute(f"INSERT INTO industry (industry) VALUES ('{industry_name}')")
            conn.commit()
        industry_id = conn.execute(f"SELECT industry_id FROM industry WHERE industry = '{industry_name}'")
        industry_id = industry_id.fetchall()
        self.industry_id = industry_id[0]   # not quite sure if this is how this should work.

    def check_sector(self, sector, conn):
        '''This method takes the sector given as an input and checks to see if it exists in the database. If it does not already exist in the database, it adds it to the database.'''

        sectors = conn.execute('SELECT sector FROM sector')
        sectors = sectors.fetchall()
        
        # loop thru all of the different sectors in the sector table
        # if the sector is found, we exit the while loop w/o adding it.
        # if the sector is not found, we add it to the db
        x = False
        for i in sectors:
            if i == sectors:
                x = True
                break
        if x is False:
            conn.execute(f"INSERT INTO sector (sector) VALUES ('{sector}')")
            conn.commit()
        sector_id = conn.execute(f"SELECT sector_id FROM sector WHERE sector = '{sector}'")
        sector_id = sector_id.fetchall()
        self.sector_id = sector_id[0]

class Price:

    def __init__(self, stock, open_price, high_price, low_price, close_price, volume, price_datetime, dividends=None, stock_splits=None):
        self.stock = stock  # this should be a stock object
        self.open_price = open_price
        self.high_price = high_price
        self.low_price = low_price
        self.close_price = close_price
        self.volume = volume
        self.price_datetime = price_datetime
        # optional
        self.dividends = dividends
        self.stock_splits = stock_splits

def get_current_datetime():
    '''Determines the current Datetime in Year-Month-Day Hour:Min:Sec format and returns it as a string.'''
    current_datetime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S"')
    return current_datetime

