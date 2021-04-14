# Changelog for in-progress branch

## 04/13/2021 22:14
Pi crashed on an error during the API call. Looks like it may be a problematic symbol in my data that has a slash in it. Not quite sure what happened there. Added a try/except block for the API call in order to skip over problematic symbols such as that one. 

## 04/13/2021 07:43
Script crashed last night on symbol CEL due to no info on the symbol (possible delisting). Added in a few lines to deal w/ symbols w/ no data present (by skipping them). Commit/push and going to attempt to keep running on Raspberry Pi throughout the day.

## 04/12/2021 22:24
Rewrote my get_price_history_loop.py file to use pandas to organize the data and write it to the database, while also using sklearn to deal with NaN values. The code ran past my problem symbol. Going to attempt running on Raspberry Pi overnight.

## 04/12/2021 21:37
Figured out how to write my problematic symbol (BEBE) to my database. A few issues were present: some apparent formatting issues in my sqlite query, and some NaN values in the data. This was corrected by taking the data from yfinance in its original dataframe form, changing column names as needed for my database, adding the necessary information, and dealing with NaN values with sklearn's KNNImputer. I am now going to heavily rewrite the get_price_history_loop to implement these changes.

I will also soon need to stop pushing my database to github b/c it's going to get too big. Will need to figure out something else when this happens.

## 04/12/2021 20:32
I think I made some progress regarding my crashes issue? For some reason an extra column is getting added where it shouldn't be. Going to try just taking the values as a pandas df and just appending it to the sqlite3 table. Will first attempt to do this in the sketches notebook.

## 04/11/2021 22:38
Ran into an issue w/ a symbol when running on the Raspberry Pi. Since the symbol is over 200 entries in, I didn't want to watch my laptop for that long, so I added a feature that allows the script to continue from where it left off when it stops or crashes (using a pickle). I'm planning on using the pickle created right before the crash to do some additional debugging on my actual desktop or laptop, which is a bit easier than doing it on a Raspberry Pi accessed via VNC. NOTE: Need to figure out a better way to move the database file around. It will quickly outgrow what github will handle.

## 04/11/2021 18:08
The get_price_history_loop.py file is working now, although I haven't tested to see how it handles problematic data fetched from yfinance (or no data). Going to commit/push, then attempt to run this on my Raspberry Pi overnight to see when it crashes, or if it is robust enough to leave running for now to collect data.

## 04/11/2021 16:14
Code for the get_price_history_loop file not yet test-run. Changes I'd like to add before finalizing it are below.

Still needed:
    - While loop, so it can run indefinetly on a Raspberry Pi
    - Possible log file to keep track of the loop.

## 04/11/2021 15:09
Got a code cell in my sketches notebook to add all of the securities I'd like to collect data for to my SQLite database. Ran that, now working on creating the script that will run indefinetly collecting data (which for me, will be running on a Raspberry Pi)

## 04/08/2021 07:47
Running into syntax errors w/ my SQL queries. Suspect it has something to do w/ how the check_industry and sector methods are returning the sector/industry ID, which is as a tuple. Need to sort that out.

## 04/07/2021 22:38
Working on building the script to take all of the symbols in a list and fetch the data for those symbols. Attempting to make a slightly OOP approach, partly for practice, and partly to keep things neat and tidy. Unsure if my check_industry and check_sector methods work or not. Goal is to update the check_sector method to actually get it to assign a value to self.sector_id, then to finish building the loop and actually try to run it.

## 04/06/2021 07:48
Returning to this project after a long break, which is largely because I had a lot of other learning to do before going much further with this one. I have now started the freeCodeCamp Tensorflow course, so I feel like I will actually be able to start making progress with this one soon. Changes to be implemented are listed below:

- Switch from MySQL to SQLite
    - MySQL is being difficult on Raspberry Pi w/ USB storage, so to simplify things, I will be switching to a SQLite database. I will likely also not at first be using all of the extra data I had accumulated the first time, and will be relying mainly on yfinance.
- The data gathering script will run ad infinitum, so the project will be ready whenever I am able to actually start training. 

New changes are being drafted in the sketches.ipynb notebook.