from matplotlib.pyplot import get
from predict_symbol import predict_symbol
import pickle
import datetime
from reddit_calls import get_new_symbols
import pandas as pd

# starting vars
pickle_symbol_read = False
after_hours_complete = False
stats_dict = {}

# load the list of all symbols (loads once)
with open('all_symbols.txt', 'r') as f:
    f = f.readlines()
    symbols = [i.replace("\n", "") for i in f]

# loop through all symbols that are worth examining
while True:

    try:
        infile = open('all_symbol_loop', 'rb')
        pickle_symbol = pickle.load(infile)
        pickle_symbol_read = False
    except:
        pickle_symbol = None
        pickle_symbol_read = True

    for symbol in symbols:
        
        if pickle_symbol_read is False:
            if symbol == pickle_symbol:
                pickle_symbol_read = True
            else:
                continue

        # if current time is after market hours, check watchlist symbols post market close
        if datetime.datetime.now().time() > datetime.datetime(2021, 7, 11, 17, 00, 00).time():
            if after_hours_complete is True:
                continue
            with open('watchlist_symbols.txt', 'r') as f:
                watchlist = f.readlines()
                watchlist = [i.replace("\n", "") for i in f]

            # iterate over the watchlist and create predictions, saving charts, saving info as csv
            for symbol in watchlist:
                stats_dict[symbol] = predict_symbol(symbol, show_downward_predictions=True, save_prediction_as_csv=True, is_watchlist_symbol=True)
                # comment out continue below when testing is done, uncomment out above line
                # continue

            # get new symbols from reddit, if there's anything interesting
            new_symbols = pd.Series(get_new_symbols())
            new_symbols = new_symbols.sort_values(ascending=False)

            # predict on new symbols from reddit
            for reddit_symbol in new_symbols:
                stats_dict[reddit_symbol] = predict_symbol(reddit_symbol, show_downward_predictions=True, save_prediction_as_csv=True, is_watchlist_symbol=True)

            # insert send email module here

        # predict all symbols
        stats_dict[symbol] = predict_symbol(symbol, min_percent_increase=20)