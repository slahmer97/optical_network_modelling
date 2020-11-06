from tensorflow.keras.optimizers import Adam

import problem
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input

from tensorflow.keras import backend as K


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


class Regressor():
    def __init__(self, input_size=56, output_size=32):
        self.network = Regressor.get_model(input_size, output_size)
        self.network.summary()

    @staticmethod
    def get_model(shape, output_size):

        dropout = 0.25
        model = Sequential()
        model.add(Input(shape=shape))
        model.add(Dense(64, activation='tanh'))
        model.add(Dense(128, activation='tanh'))
        model.add(Dense(128, activation='tanh'))
        model.add(Dropout(dropout))
        model.add(Dense(128, activation='tanh'))
        model.add(Dropout(dropout))
        model.add(Dense(128, activation='tanh'))

        model.add(Dropout(dropout))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(output_size, ))

        opt = keras.optimizers.Adam(lr=0.001)
        rmse = tf.metrics.RootMeanSquaredError()
        model.compile(loss="mean_squared_error", optimizer=opt, metrics=[rmse])
        return model

    @staticmethod
    def process_input(x_, slice_=True):
        input_ = x_[0]
        ret_vec = np.empty([24], dtype=np.float64)
        i = 0
        # print("len : {}".format(len(input_)))
        if slice_:
            slice_num = 8 - len(input_) + 1
        else:
            slice_num = 1

        for elm in input_:
            if elm[0] == 'EDFA':
                ret_vec[i] = 1
            if elm[0] == 'SMF':
                ret_vec[i] = -1
            ret_vec[i + 1] = elm[1][0]
            ret_vec[i + 2] = elm[1][1]
            i += 3

        for j in range(i, 24):
            ret_vec[j] = 0.0
        res = []
        for i in range(0, slice_num):
            tmp = np.roll(ret_vec, 3 * i)
            res.append(np.concatenate((x_[1], tmp), axis=0).reshape(56))
        return res

    @staticmethod
    def get_ds(X_, Y_, yes=True):
        X = np.empty(shape=(X_.shape[0] * 8, 56), dtype=np.float64)
        if yes:
            Y = np.empty(shape=(Y_.shape[0] * 8, 32), dtype=np.float64)
        else:
            Y = None
        i = 0
        j = 0
        for training_example in X_:
            # X[i] = Regressor.process_input(training_example)
            # i += 1
            if yes:
                ret = Regressor.process_input(training_example)
            else:
                ret = Regressor.process_input(training_example, slice_=False)

            for elm in ret:
                X[i] = elm
                i = i + 1
                if Y_ is not None:
                    Y[i] = Y_[j]

            j = j + 1

        print("i = {}".format(i))
        print("before shape : {}".format(X.shape))
        X = np.resize(X, (i, 56))
        if Y_ is not None:
            Y = np.resize(Y, (i, 32))
        print("after resize shape : {}".format(X.shape))
        X = X.reshape((i, 56))
        print("after reshape shape : {}".format(X.shape))

        return X, Y

    def fit(self, X_, Y_):
        X, Y = Regressor.get_ds(X_, Y_)
        scaler = MinMaxScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        # X_val, Y_val = Regressor.get_ds(X_, Y_)
        # print(X.shape)
        # print(Y.shape)
        # validation_data=(X_val, Y_val)

        self.network.fit(X, Y, batch_size=512, epochs=100, verbose=1,
                         validation_split=0.33)

    def save(self):
        self.network.save("model.h5")

    def load(self):
        try:
            self.network = keras.models.load_model('model.h5')
            self.network.summary()
        except Exception as inst:
            print("No previous model to load exception : {}".format(inst))

    def evaluate(self, XTest, YTest):
        X, Y = Regressor.get_ds(XTest, YTest)
        scores = self.network.evaluate(X, Y, verbose=1)
        print("%s: %.2f%%" % (self.network.metrics_names[1], scores[1] * 100))

    def predict(self, X_):
        print("\n\npredict Shape {}\n\n\n\h".format(X_.shape))
        (X, Y) = Regressor.get_ds(X_, None, yes=False)
        print("X shape : {}".format(X.shape))
        return self.network.predict(X)
