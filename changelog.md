# Changelog for in-progress branch

## 05/23/2021 20:54
Added in a few code cells that contain the necessary code to filter out the weird time values. Will organize a bit more in the next commit.

## 05/22/2021 21:39
Attempted training a model w/ 20 epochs. Running into an issue where the imputing creates massive spikes in the data, need to attempt to adjust the imputer.

### 23:13
I distinctly remember writing code that filtered out the weird times (not an even 15 minute interval), but for some reason, that code is now gone and not a part of any of my prior commits. Going to have to try to rewrite it unfortunately, as that is what seems to be causing a lot of the 'spikiness' in my data, which is throwing off my model/predictions.

## 05/18/2021 08:49
A lot of cleanup left to do with this notebook. Made it to the end of the tutorial, but my results are wildly off. Not sure if my dataset is just that much more limited or if something else is the issue. I may try redoing this using end of day closing prices instead of 15 min. intraday prices.

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