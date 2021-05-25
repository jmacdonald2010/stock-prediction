# Changelog for EOD-price-gathering-01

## Goal
Goal is to update the get_price_history_loop.py script to not only get 15 min. intraday history, but also EOD history going back indefinetly. It will take a long time before I have enough intraday 15-min. price history to make any accurate predictions, so I will use EOD prices in the meantime.

## 05/25 19:45
Fixed a couple of bugs in the code that were writing all entries to the regular price_history table and also creating massive amounts of redundant data. Going to place on an old laptop and allow to run for a few days to see if I get better data. 

## 05/18/2021 09:35
Started the notebook. More work to be done on it. Commit prior to going to work.

### 21:56
Basic functionality almost works. Placing it all into an actual .py file for easier debugging.

### 22:19
Basic functionality seems to work. Going to try running it on a raspberry pi overnight and throughout the day to see if it crashes. Will merge once confirmed successful.