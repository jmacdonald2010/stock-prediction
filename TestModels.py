from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import ConvLSTM2D

class TestModels:

    def __init__(self, n_steps_in, n_steps_out, units=100, n_features=1):
        self.n_steps_in = n_steps_in
        self.n_steps_out = n_steps_out
        self.units = units
        self.n_features = n_features
        self.models = ['stackedLSTM', 'bidirectionalLSTM', 'CNNLSTM', 'ConvLSTM']