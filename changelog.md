# Changelog for in-progress branch

## 05/18/2021 08:18
Deleted a bunch of cells to tidy up. Still running into errors in the prediction part of the notebook due to array shapes.

## 05/17/2021 22:30
I think I know what I need to do to get this to work properly. Need to in the cell where the minmax objects are defined, place the symbol I want to predict as the single one, and DON'T add that one to the regular minmax object. Go from here w/ training, predictions, etc. Hopefully.

## 05/16/2021 12:56
Going to change the way some things are set up for the actual ML code. My data is slightly different now (using the clean_df dataframe for the ML dataset). Committing just in case I mess it up quite badly.

### 14:34
Model trains, but I need to fix up some things that are causing problems w/ predictions, which I think comes down to the minmax_single variable not actually being used.

## 05/14/2021
Working on building the code for the ML model. Running into issues w/ my data. A lot of inconsistencies in it, need to spend some time filtering out the NaN values. The imputing code I used when storing things into the database doesn't seem to have done the trick. Will try playing with the merge, and then dropping tickers w/ too many NaNs.