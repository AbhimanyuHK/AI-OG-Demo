# LSTM for international airline passengers problem with regression framing
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential, load_model

# fix random seed for reproducibility
tf.random.set_seed(7)

DATA_FILE = "data/ml_data.h5"
ML_FILE = 'ml_models/forecasting_with_tensorflow_1.h5'


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


def load_ml_model():
    return load_model(ML_FILE)


def load_ML_data_file():
    return load_model(DATA_FILE)


class ModelData:
    DATASET = np.array([])
    TRAIN_PREDICTION = np.array([])
    TEST_PREDICTION = np.array([])

    PLOT_TRAIN_DATA = None
    PLOT_TEST_DATA = None

    TEST_RMSE = 0
    TRAIN_RMSE = 0

    @staticmethod
    def to_df():
        df = pd.DataFrame()
        df["Date"] = ModelData.DATASET[:, 0]
        df["Store"] = ModelData.DATASET[:, 1]
        df["Actual"] = ModelData.DATASET[:, 2]
        df["Train"] = ModelData.PLOT_TRAIN_DATA[:, 2]
        df["Test"] = ModelData.PLOT_TEST_DATA[:, 2]
        return df


class DemandForecastingModel(metaclass=Singleton):
    DATASET = np.array([])
    TRAIN_PREDICTION = np.array([])
    TEST_PREDICTION = np.array([])

    def __init__(self):
        self.dataset = np.array([])
        self.look_back = 1

        self.min_max_scaler = MinMaxScaler(feature_range=(0, 1))
        self.data_split_index = 0

        # create and fit the LSTM network
        self.model = Sequential()
        self.model.add(LSTM(4, input_shape=(1, self.look_back)))
        self.model.add(Dense(1))
        self.model.compile(loss='mean_squared_error', optimizer='adam')

        # self.main()

    def main(self):

        self.read_data()
        self.normalize_the_dataset()
        train, test = self.__split_data_train_and_test__()

        train_X, train_Y = self.create_dataset(train)
        test_X, test_Y = self.create_dataset(test)

        train_X = self.reshape_data(train_X)
        test_X = self.reshape_data(test_X)

        self.fit(train_X, train_Y)

        self.train_validate(train_X, train_Y)
        self.test_validate(test_X, test_Y)

        self.save_model()
        ModelData.DATASET = self.min_max_scaler.inverse_transform(self.dataset)
        self.plot_data()

    def predict(self, store_option, date_option):
        _input_array = np.array([[date_option, store_option, 0.0]])
        _input_array_sc = self.min_max_scaler.transform(_input_array)
        _input_array_sc_d = np.array([_input_array_sc[0:1]])
        _input_array_sc_d_r = self.reshape_data(_input_array_sc_d)
        result = self.model.predict(_input_array_sc_d_r)
        final_result = self.min_max_scaler.inverse_transform(
            self._inverse_transform_predict_(_input_array_sc_d_r, predict_data=result)
        )
        df = pd.DataFrame(final_result, columns=['Date', 'Store', 'Predicted Demand'])
        return df

    # convert an array of values into a dataset matrix
    def create_dataset(self, _dataset):
        dataX, dataY = [], []
        for i in range(len(_dataset) - self.look_back - 1):
            a = _dataset[i:(i + self.look_back)]
            b = _dataset[i + self.look_back, 2]
            dataX.append(a)
            dataY.append(b)
        return np.array(dataX), np.array(dataY)

    def read_data(self, file='data/final_demand_invt.csv'):
        # load the dataset
        dataframe = read_csv(file, usecols=[0, 1, 2], engine='python')
        dataframe = dataframe[-100:]
        print(dataframe)
        self.dataset = dataframe.values
        self.dataset = self.dataset.astype('float32')
        return self.dataset

    def normalize_the_dataset(self):
        self.dataset = self.min_max_scaler.fit_transform(self.dataset)

    def __split_data_train_and_test__(self):
        train_size = int(len(self.dataset) * 0.67)
        train, test = self.dataset[0:train_size, :], self.dataset[train_size:len(self.dataset), :]
        return train, test

    def reshape_data(self, dataset):
        print(dataset.shape, "=>")
        print((dataset.shape[0], 3, dataset.shape[1]))
        return np.reshape(dataset, (dataset.shape[0], 3, dataset.shape[1]))

    def reshape_test_data(self, testX):
        return np.reshape(testX, (testX.shape[0], 3, testX.shape[1]))

    def fit(self, train_X, train_Y):
        self.model.fit([train_X], train_Y, epochs=20, batch_size=1, verbose=2)

    def validate(self, train_X, test_y, test_X, test_Y):
        self.train_validate(train_X, test_y)
        self.test_validate(test_X, test_Y)

    def test_validate(self, test_X, test_Y):
        predicted_y = self.model.predict(test_X)
        self.test_inverse_transform(test_X, test_Y, predicted_y)

    def train_validate(self, train_X, test_y):
        train_predict = self.model.predict(train_X)
        self.train_inverse_transform(train_X, test_y, train_predict)

    def _inverse_transform_predict_(self, test_data, predict_data=np.array([]), test_y_data=np.array([])):
        x_predict = test_data.copy()
        if np.any(predict_data):
            x_predict[:, 2] = predict_data
        if np.any(test_y_data):
            x_predict[:, 2] = test_y_data.reshape(test_y_data.shape[0], 1)

        return x_predict.reshape((x_predict.shape[0] * x_predict.shape[2]), x_predict.shape[1])

    def train_inverse_transform(self, train_X, train_Y, train_predict):
        train_predict_in = self.min_max_scaler.inverse_transform(
            self._inverse_transform_predict_(train_X, predict_data=train_predict)
        )
        train_y_in = self.min_max_scaler.inverse_transform(
            self._inverse_transform_predict_(train_X, test_y_data=train_Y)
        )
        ModelData.TRAIN_RMSE = self.rmse_score(train_y_in, train_predict_in, "Train")
        ModelData.TRAIN_PREDICTION = train_predict_in
        return train_predict_in, train_y_in

    def test_inverse_transform(self, test_X, test_Y, test_predict):
        test_predict_in = self.min_max_scaler.inverse_transform(
            self._inverse_transform_predict_(test_X, predict_data=test_predict)
        )
        test_y_in = self.min_max_scaler.inverse_transform(
            self._inverse_transform_predict_(test_X, test_y_data=test_Y)
        )
        ModelData.TEST_RMSE = self.rmse_score(test_y_in, test_predict_in, "Test")
        ModelData.TEST_PREDICTION = test_predict_in
        return test_predict_in, test_y_in

    @classmethod
    def rmse_score(cls, test_y, predicted_y, message):
        # calculate root mean squared error
        score = np.sqrt(mean_squared_error(test_y[:, 2], predicted_y[:, 2]))
        print(f'{message} Score: {score} RMSE')
        return score

    def save_model(self):
        pass
        # self.model.save(ML_FILE)

    def plot_data(self):
        # shift train predictions for plotting
        trainPredictPlot = np.empty_like(self.dataset)
        trainPredictPlot[:, :] = np.nan
        trainPredictPlot[
        self.look_back:len(ModelData.TRAIN_PREDICTION) + self.look_back, :
        ] = ModelData.TRAIN_PREDICTION
        ModelData.PLOT_TRAIN_DATA = trainPredictPlot

        # shift test predictions for plotting
        testPredictPlot = np.empty_like(self.dataset)
        testPredictPlot[:, :] = np.nan
        testPredictPlot[
        len(ModelData.TRAIN_PREDICTION) + (self.look_back * 2) + 1:len(self.dataset) - 1, :
        ] = ModelData.TEST_PREDICTION
        ModelData.PLOT_TEST_DATA = testPredictPlot

        # with open(DATA_FILE, 'ab') as f:
        #     pickle.dump(ModelData, f)

# # load the dataset
# dataframe = read_csv('data/demand.csv', usecols=[0, 1, 4], engine='python')
# dataframe = dataframe[:1000]
# print(dataframe)
# dataset = dataframe.values
# dataset = dataset.astype('float32')
#
# # normalize the dataset
# scaler = MinMaxScaler(feature_range=(0, 1))
# dataset = scaler.fit_transform(dataset)
# print(dataset)

# # split into train and test sets
# train_size = int(len(dataset) * 0.67)
# test_size = len(dataset) - train_size
# train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

# # reshape into X=t and Y=t+1
# look_back = 1
# trainX, trainY = create_dataset(train, look_back)
# testX, testY = create_dataset(test, look_back)

# # reshape input to be [samples, time steps, features]
# trainX = np.reshape(trainX, (trainX.shape[0], 3, trainX.shape[1]))
# testX = np.reshape(testX, (testX.shape[0], 3, testX.shape[1]))

# # create and fit the LSTM network
# model = Sequential()
# model.add(LSTM(4, input_shape=(1, look_back)))
# model.add(Dense(1))
# model.compile(loss='mean_squared_error', optimizer='adam')
# model.fit([trainX], trainY, epochs=20, batch_size=1, verbose=2)

# # make predictions
# _trainPredict = model.predict(trainX)
# _testPredict = model.predict(testX)
# print(trainX.shape, _trainPredict.shape)


# print(trainX)
# print(trainPredict)

# def _inverse_transform_predict_(test_data, predict_data=np.array([]), test_y_data=np.array([])):
#     x_predict = test_data.copy()
#     if np.any(predict_data):
#         x_predict[:, 2] = predict_data
#     if np.any(test_y_data):
#         x_predict[:, 2] = test_y_data.reshape(test_y_data.shape[0], 1)
#
#     return x_predict.reshape((x_predict.shape[0] * x_predict.shape[2]), x_predict.shape[1])


# trainPredict = trainX.copy()
# trainPredict[:, 2] = _trainPredict
# trainPredict = trainPredict.reshape((trainPredict.shape[0] * trainPredict.shape[2]), trainPredict.shape[1])
#
# testPredict = testX.copy()
# testPredict[:, 2] = _testPredict

# # invert predictions
# trainPredict = scaler.inverse_transform(_inverse_transform_predict_(trainX, predict_data=_trainPredict))
# trainY = scaler.inverse_transform(_inverse_transform_predict_(trainX, test_y_data=trainY))
# testPredict = scaler.inverse_transform(_inverse_transform_predict_(testX, predict_data=_testPredict))
# testY = scaler.inverse_transform(_inverse_transform_predict_(testX, test_y_data=testY))
#
# # calculate root mean squared error
# trainScore = np.sqrt(mean_squared_error(trainY[:, 2], trainPredict[:, 2]))
# print('Train Score: %.2f RMSE' % (trainScore))
# testScore = np.sqrt(mean_squared_error(testY[:, 2], testPredict[:, 2]))
# print('Test Score: %.2f RMSE' % (testScore))
#
# model.save(f'ml_models/forecasting_with_tensorflow_RMSE_{testScore}.h5')


# # shift train predictions for plotting
# trainPredictPlot = np.empty_like(dataset)
# trainPredictPlot[:, :] = np.nan
# trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict
#
# # shift test predictions for plotting
# testPredictPlot = np.empty_like(dataset)
# testPredictPlot[:, :] = np.nan
# testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1, :] = testPredict
#
# # plot baseline and predictions
# rev_dataset = scaler.inverse_transform(dataset)
# plt.plot(rev_dataset[:, 0], rev_dataset[:, 2])
# plt.plot(trainPredictPlot[:, 0], trainPredictPlot[:, 2])
# plt.plot(testPredictPlot[:, 0], testPredictPlot[:, 2])
# plt.show()

# DemandForecastingModel().main()
