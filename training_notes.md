# Training notes

## Description

Code blocks for training are recorded here, along with my observations.

### 05/31/2021 14:35
```
LSTM architecture
regressor = tf.keras.Sequential()

First layer, w/ dropout regularization
regressor.add(tf.keras.layers.LSTM(units=20, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
regressor.add(tf.keras.layers.Dropout(0.3))

Second Layer
regressor.add(tf.keras.layers.LSTM(units=20))
regressor.add(tf.keras.layers.Dropout(0.5))

Output layer
regressor.add(tf.keras.layers.Dense(units=1))

Compile LSTM
regressor.compile(optimizer='adam', loss='mean_squared_error')

Fit to Training set
fit to training set
num_features = len(training_data.columns)
progress = 1
for i in training_data.columns:
    print("Fitting to", i)
    print("Training feature", progress, "of", num_features)
    regressor.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
    progress += 1
```

#### Notes:

Train/Test split: 90/10

Overall trend info is correct, although the predicted values for FCEL are higher than what they actually were. This is likely due to large price spike that occured (and was still falling) around the 90/10 split. 

### 05/31/2021 14:41

```
LSTM architecture
regressor = tf.keras.Sequential()

First layer, w/ dropout regularization
regressor.add(tf.keras.layers.LSTM(units=20, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
regressor.add(tf.keras.layers.Dropout(0.2))

Second Layer
regressor.add(tf.keras.layers.LSTM(units=20, return_sequences=True))
regressor.add(tf.keras.layers.Dropout(0.2))

Third
regressor.add(tf.keras.layers.LSTM(units=20))
regressor.add(tf.keras.layers.Dropout(0.5))

Output layer
regressor.add(tf.keras.layers.Dense(units=1))

Compile LSTM
regressor.compile(optimizer='adam', loss='mean_squared_error')

Fit to Training set
fit to training set
num_features = len(training_data.columns)
progress = 1
for i in training_data.columns:
    print("Fitting to", i)
    print("Training feature", progress, "of", num_features)
    regressor.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
    progress += 1
```

#### Notes:

Train/Test split: 90/10
Layers, excluding output: 3
Notes: THis was actually worse than the model with less layers. This one placed the prices remarkably higher than the previous version. 

### 05/31/2021 14:46

```
# LSTM architecture
regressor = tf.keras.Sequential()

# First layer, w/ dropout regularization
regressor.add(tf.keras.layers.LSTM(units=20, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
regressor.add(tf.keras.layers.Dropout(0.2))

# Second
regressor.add(tf.keras.layers.LSTM(units=20))
regressor.add(tf.keras.layers.Dropout(0.5))

# Output layer
regressor.add(tf.keras.layers.Dense(units=1))

# Compile LSTM
regressor.compile(optimizer='adam', loss='mean_squared_error')

# Fit to Training set
# fit to training set
num_features = len(training_data.columns)
progress = 1
for i in training_data.columns:
    print("Fitting to", i)
    print("Training feature", progress, "of", num_features)
    regressor.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
    progress += 1
```

#### Notes:
- 90/10 train test split.
- Layers: 2
- Date Range: 2019 - Present.
- Epochs changed to 20 from 10. Resulting prediction on FCEL is actually best so far. Is slightly lower than the real data, but is much closer in the descending price of FCEL. Need to try with some other symbols.