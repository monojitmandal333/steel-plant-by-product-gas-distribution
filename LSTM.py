import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
    
# LSTM Model Building
class LSTMTrend():
    def __init__(self,n_epoch,n_batch,n_neurons):
        self.__n_epoch = n_epoch
        self.__n_batch = n_batch
        self.__n_neurons = n_neurons
        self.__model = Sequential()
    def fit(self,X,y):
        lstm = self.__model
        X = X.reshape((X.shape[0], 1, X.shape[1]))
        lstm.add(LSTM(self.__n_neurons,return_sequences = True, input_shape=(X.shape[1], X.shape[2])))
        lstm.add(Dropout(0.2))
        lstm.add(LSTM(self.__n_neurons,return_sequences = False))
        lstm.add(Dropout(0.2))
        lstm.add(Dense(y.shape[1]))
        lstm.compile(loss='mean_squared_error', optimizer='adam')
        lstm.fit(X, y, epochs=self.__n_epoch, verbose=0)
        self.__model = lstm
    def predict(self,X):
        X = X.reshape((X.shape[0], 1, X.shape[1]))
        y = self.__model.predict(X)
        return y