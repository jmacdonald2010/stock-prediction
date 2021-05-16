# Changelog for in-progress branch

## 05/16/2021 12:56
Going to change the way some things are set up for the actual ML code. My data is slightly different now (using the clean_df dataframe for the ML dataset). Committing just in case I mess it up quite badly.

## 05/14/2021
Working on building the code for the ML model. Running into issues w/ my data. A lot of inconsistencies in it, need to spend some time filtering out the NaN values. The imputing code I used when storing things into the database doesn't seem to have done the trick. Will try playing with the merge, and then dropping tickers w/ too many NaNs.