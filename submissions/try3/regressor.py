import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


def create_chunks(list_name, n):
    for i in range(0, len(list_name), n):
        yield list_name[i:i + n]


def process_data(X_train, Y_train=None, shifting=False):
    X_P30 = []
    Y_P30 = []
    params = []
    for i in range(0, len(X_train)):
        example_train = X_train[i]
        applied_modules = example_train[0]
        p_in_32 = example_train[1]
        if np.all(p_in_32 == 0):
            continue
        length = len(applied_modules)

        X_P30.append(np.array(p_in_32).reshape((1, 32)))
        i_params = []
        for k in range(0, 8):
            if k < length:
                mod = applied_modules[k]
                if mod[0] == 'EDFA':
                    mod_id = 1
                else:
                    mod_id = -1
                param1 = mod[1][0]
                param2 = mod[1][1]
            else:
                mod_id = 0
                param1 = 0
                param2 = 0
            i_params.extend([mod_id, param1, param2])
        params.append(i_params)
        from collections import deque
        if shifting:
            Y_P30.append(np.array(Y_train[i]).reshape((1, 32)))
            new_tmp = deque(i_params)

            for _ in range(length, 8):
                new_tmp.rotate(3)
                X_P30.append(np.array(p_in_32).reshape((1, 32)))
                Y_P30.append(np.array(Y_train[i]).reshape((1, 32)))
                params.append(list(new_tmp))

    return np.array(X_P30), np.array(params), np.array(Y_P30)


class Regressor:
    def __init__(self):
        import numpy as np
        np.random.seed(1337)
        self.last_ac = None
        self.last_vac = None
        self.saved_weights = {}
        self.batch_size = 256
        self.epochs_num = 100
        self.vector_input1 = keras.Input(shape=(32,), name="R30_input_1")

        self.hidden_left_0 = layers.Dense(128, name="hidden_left_0", activation="linear")(self.vector_input1)
        self.hidden_left_0 = layers.Dense(512, name="hidden_left_1", activation="linear")(self.hidden_left_0)
        self.hidden_left_1_1 = layers.Dense(512, name="hidden_left_1_1", activation="tanh")(self.hidden_left_0)
        self.hidden_left_2 = layers.Dense(512, name="hidden_left_2", activation="tanh")(self.hidden_left_1_1)
        self.hidden_left_3 = layers.Dense(512, name="hidden_left_3", activation="tanh")(self.hidden_left_2)

        self.params_input = keras.Input(shape=(8 * 3,), name="params_input")
        self.hidden_right_0 = layers.Dense(128, name="hidden_right_0", activation="linear")(self.params_input)
        self.hidden_right_5 = layers.Dense(128, name="hidden_right_5", activation="linear")(self.hidden_right_0)

        self.concat = layers.concatenate([self.hidden_right_5, self.hidden_left_3])
        self.hidden_middle_1 = layers.Dense(128, name="hidden_middle_1", activation="linear")(self.concat)
        self.hidden_middle_5 = layers.Dense(128, name="hidden_middle_5", activation="linear")(self.hidden_middle_1)

        self.output_layer = layers.Dense(32, name="output_layer", activation="relu")(self.hidden_middle_5)
        self.model = keras.Model(
            inputs=[self.vector_input1, self.params_input],
            outputs=[self.output_layer],
        )
        opt = tf.optimizers.Adam(lr=0.0001)
        rms = tf.keras.metrics.RootMeanSquaredError()
        mse = tf.keras.losses.MeanSquaredError()
        self.model.compile(loss=mse, optimizer=opt, metrics=[rms])
        self.model.summary()

    def predict(self, X):
        X_P30, Params, _ = process_data(X)

        return self.model.predict(
            x=
            {
                "R30_input_1": np.array(X_P30).reshape(len(X_P30), 32),
                "params_input": np.array(Params).reshape((len(Params), 3 * 8))
            }
        )

    def evaluate(self, X, Y):
        pass

    def fit(self, X_train, Y_train):
        X_P30, Params, Y_P30 = process_data(X_train, Y_train, shifting=True)
        print("Total len : {}".format(len(X_P30)))
        try:
            self.model.fit(
                x=
                {
                    "R30_input_1": np.array(X_P30).reshape(len(X_P30), 32),
                    "params_input": np.array(Params).reshape((len(Params), 3 * 8)),
                },
                y=np.array(Y_P30).reshape((len(Y_P30), 32)),
                epochs=self.epochs_num, batch_size=self.batch_size,
            )
        except Exception as inst:
            print("exception : {}".format(inst))