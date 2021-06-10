# Changelog for in-progress branch

## 06/10/2021 16:12

For future prediction cells: adds new datetime values to the index for the next 30 days.

## 06/09/2021 06:31

Model w/ additional data has finished training. Unsure of its accuracy yet, has not been tested. Commit/push prior to work.

## 06/07/2021 08:46

New model finished training using a batch size of 2. Some MSE values were better, some slightly worse than with 4, but the movement of the predicted prices is largely static, so I wouldn't consider it to be terribly reliable. So far 20210606_153456 is the most accurate, and I may use this for potential predictions soon.

### To Do:

Clean up the notebook, or make a new one that does what it needs to do.

## 06/06/2021 18:23

Ran a few runs of the automated script. So far, it looks like 100 epochs starting from 2019 to present provides the lowest MSE for a selection of ticker symbols. Will likely continue to try training with 100 epochs, but may also attempt to get official data on smaller amounts of epochs (1-10). Will attempt to work with how many layers I'm using along with altering other settings.

### 16:34 (prev. time issues w/ my computer)

Overall best MSE results are now w/ the model for 20210606_153456. 2020 to present, 100 units instead of 20. 

### 22:34
Using a batch size of 4 has created the most accurate model so far. Took six hours to train. May try another one that is an even smaller batch size, just to see if it's any more accurate. This model is model_20210606_163712. 

## 06/02/2021 07:55
Made an absolute mess of my EOD notebook, but that is due to attempting to train models and increase accuracy. Going to write a script that will detail how everything is, save charts, MSE values, and model parameters.

### 08:23 

Started working on a loop to create plots and evaluate MSE values.

### 20:26
Finished writing the script that runs the whole notebook, trains a model, and predicts some symbols. Looks like 100 epochs is more accurate than 200.

## 05/31/2021 12:24
Running into errors in the end, when inverting the transformed data back into a normal readable format. Not sure why this error is occuring.

### 12:40
Sorted out the 12:24 error. Prediction is now full of NaN's. Likely due to NaNs in training data, which need resolved. Investigating this error.

### 12:59
Resolved the above error. Now able to predict symbols. Now need to work on finetuning the model and all of that.

### 14:52
Predictions on FCEL seem to be working pretty well. Going to rewrite some code to make it easier to try this with other symbols.

### 14:57
Adjusted some code so you can easily plug your symbol of choice in somewhere and go from there.

### 16:10
Overall, needs a fair bit of work. Seems to only really predict FCEL moderately accurately and only from 2019 to present. Other symbols do not fare as well.

## 05/26/2021 22:50
Started work on the eod notebook. 

## 05/25/2021 19:15
Now running into issues with my data just removing the bad intraday datetimes. No idea why this is now happening, could've sworn it was working correctly on 5/23. Going to create a new notebook to work w/ EOD values, as they may be more predictable. The mlTests02 notebook will be abandoned, and is currently an unusable mess.

### 19:32
New notebook created, realized new database is all messed up. Moving to EOD price collection branch for now.

## 05/23/2021 20:54
Added in a few code cells that contain the necessary code to filter out the weird time values. Will organize a bit more in the next commit.

### 22:00
Running into an odd issue where when I split my data into training/test it gets all messed up and inaccurate. Going to work w/ finding a better way of taking my original dataframe and creating my test/train data sets out of it.

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