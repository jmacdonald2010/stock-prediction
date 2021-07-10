# Changelog for in-progress branch

## 07/10/2021 15:13
Fixed font size issue.

## 07/07/2021 17:53
Adding in the option to not predict on falling symbols, and to set a minimum percent increase to predict on. Need to fix an issue where the the title on the predicted graphs are now too long to see everything.

## 07/06/2021 16:09
Wrote a function that trains a model on a per symbol basis (rebuilds/trains a model for each security). So far, this has given me the best results (for MSE values) on a per-symbol basis. Going to implement some changes and write the function into its own file.

### 16:56
Function now exits early w/o predictions if the predictions contain values that are below 0.

### 21:50
Using normalized MSE instead of calculating it based on the regular values. Hopefully this will help remove predictions of low dollar symbols with wild fluctuation.

## 06/29/2021 19:48
The single_symbol file is in a place that seems like it mostly works. Also working on a new script that will retrain and find the best n_steps_in to use. The optimization algorithm for that is still in progress.

## 06/28/2021 20:12
Tried training just one symbol at a time. Changed a notebook/python script file (uses VS Code's jupyter extension, # %% for new cells) to just to single-symbol prediction, and it has the best MSE overall. Going to likely use this moving forward. This also means that I can just call data from yfinance instead of keeping a database. Also will mean I may be able to do intraday.

### 21:53
Need to figure out how to properly implement the accuracy callback. Running into data shaping issues w/ my test data. 

## 06/27/2021 13:54
Looks like 50 epochs seems to do the best. Not accounting for massive random changes in price history, the model does a decent job following trends. Commit/Push prior to updating the mlTests06 notebook to predict w/o validation on the most recent data I have.

## 06/26/2021 11:37
Going back to the first ML idea I was working from (based on the mlTests03 notebook, which is since abandoned due to its messy state). Now using mlTests06 to use the original idea, which is much more accurate (likely due to a scaling issue w/ MinMaxScaler), and trains more quickly. Now trying to make it predict values beyond just the test set.

### 14:07
Finding some direction. Need to work w/ my input data and minimizing loss. Going to try changing the n_lags value from 60 to something lower (which I think works as a sort of n_steps_in value?). Eventually will change it into a script to work with minimizing loss.

## 06/19/2021 09:57
Figured out my predictions errors. Didn't realize that the n_steps_in variable is also how many data points you feed the model when attempting to predict. Right now, a lot of the predictions look the same, but also the model is barely trained. Deleted a lot of the old things from this notebook to clean it up. test_model_settings.py file is not yet ready to use, needs the fixes implemented in the mlTests05 notebook first.

## 06/18/2021 15:48
I was wrong yesterday, had not figured it out, ended up throwing a lot of methaphorical spaghetti at the wall. Realized today I only want to predict one symbol at a time using a model training on various tickers. Because of this, I am using a univariate model. Current progress is in the bottom of mlTests05. About to work on fixing the date index for the predicted dataframe. 

### 18:41
Created a test file. Going to try to leave it running endlessly on an older machine of mine, so I can use my desktop or newest laptop for other things. Will report back w/ whatever errors I encounter.

## 06/17/2021 20:30
Caused a lot of damage to my notebook, committing prior to solidifying a lot of it but I think I may have figured it out partially?

## 06/13/2021 19:44
Attempting to use a new tutorial as the basis for my machine learning model. Running into data shaping issues.

### 20:41
Got it so the data would shape properly and the model would begin to train. Getting nan loss values during training, so need to try to find the NaN values in the arrays.

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