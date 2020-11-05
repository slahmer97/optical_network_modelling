from time import sleep

import problem
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras

X_train, y_train = problem.get_train_data()
X_test, y_test = problem.get_test_data()


class Regressor():
    def __init__(self, input_size=56, output_size=32):
        self.network = Regressor.get_model(input_size, output_size)
        self.network.summary()

    @staticmethod
    def get_model(shape, output_size):
        input_layer = keras.Input(shape=shape)
        dense_layer_1 = layers.Dense(64, kernel_initializer="normal", activation='relu')(input_layer)
        dense_layer_2 = layers.Dense(512, kernel_initializer="normal", activation='relu')(dense_layer_1)
        dense_layer_3 = layers.Dense(512, kernel_initializer="normal", activation='relu')(dense_layer_2)
        dense_layer_3_1 = layers.Dense(512, kernel_initializer="normal", activation='relu')(dense_layer_3)
        dense_layer_3_3 = layers.Dense(512, kernel_initializer="normal", activation='relu')(dense_layer_3_1)
        dense_layer_4 = layers.Dense(512, kernel_initializer="normal", activation='relu')(dense_layer_3_3)
        dense_layer_5 = layers.Dense(256, kernel_initializer="normal", activation='relu')(dense_layer_4)
        dense_layer_6 = layers.Dense(32, kernel_initializer="normal", activation='relu')(dense_layer_5)
        output = layers.Dense(output_size, kernel_initializer="normal", activation=tf.nn.relu)(dense_layer_6)
        model = keras.Model(inputs=input_layer, outputs=output)
        opt = keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(loss="mean_squared_error", optimizer=opt, metrics=["mean_squared_error", 'accuracy'])
        return model

    @staticmethod
    def process_input(x_):
        input_ = x_[0]
        ret_vec = np.empty([24], dtype=np.float64)
        i = 0
        # print("len : {}".format(len(input_)))
        slice_num = 8 - len(input_) + 1

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
    def get_ds(X_, Y_):
        X = np.empty(shape=(X_.shape[0] * 8, 56), dtype=np.float64)
        Y = np.empty(shape=(Y_.shape[0] * 8, 32), dtype=np.float64)

        i = 0
        j = 0
        for training_example in X_:
            # X[i] = Regressor.process_input(training_example)
            # i += 1
            ret = Regressor.process_input(training_example)
            for elm in ret:
                X[i] = elm
                i = i + 1
                Y[i] = Y_[j]

            j = j + 1

        print("i = {}".format(i))
        print("before shape : {}".format(X.shape))
        X = np.resize(X, (i, 56))
        Y = np.resize(Y, (i, 32))
        print("after resize shape : {}".format(X.shape))
        X = X.reshape((i, 56))
        print("after reshape shape : {}".format(X.shape))

        return X, Y

    def fit(self, X_, Y_):
        X, Y = Regressor.get_ds(X_, Y_)

        # X_val, Y_val = Regressor.get_ds(X_, Y_)
        print(X.shape)
        print(Y.shape)
        # validation_data=(X_val, Y_val)
        self.network.fit(X, Y, batch_size=32, epochs=1000, verbose=1, validation_split=0.33)

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
        X = Regressor.process_input(X_)
        return self.network(X)


z = Regressor()

z.load()
print("total {}".format(len(X_train)))
sleep(2)
z.fit(X_=X_train, Y_=y_train)
z.save()
z.evaluate(X_test, y_test)
