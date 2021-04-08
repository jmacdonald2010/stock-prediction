# Changelog for in-progress branch

## 04/07/2021 22:38
Working on building the script to take all of the symbols in a list and fetch the data for those symbols. Attempting to make a slightly OOP approach, partly for practice, and partly to keep things neat and tidy. Unsure if my check_industry and check_sector methods work or not. Goal is to update the check_sector method to actually get it to assign a value to self.sector_id, then to finish building the loop and actually try to run it.

## 04/06/2021 07:48
Returning to this project after a long break, which is largely because I had a lot of other learning to do before going much further with this one. I have now started the freeCodeCamp Tensorflow course, so I feel like I will actually be able to start making progress with this one soon. Changes to be implemented are listed below:

- Switch from MySQL to SQLite
    - MySQL is being difficult on Raspberry Pi w/ USB storage, so to simplify things, I will be switching to a SQLite database. I will likely also not at first be using all of the extra data I had accumulated the first time, and will be relying mainly on yfinance.
- The data gathering script will run ad infinitum, so the project will be ready whenever I am able to actually start training. 

New changes are being drafted in the sketches.ipynb notebook.