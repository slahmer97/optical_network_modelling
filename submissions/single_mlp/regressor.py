import time

import sklearn

import problem
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


def create_chunks(list_name, n):
    for i in range(0, len(list_name), n):
        yield list_name[i:i + n]


class Regressor:
    def __init__(self):
        import numpy as np
        np.random.seed(1337)
        self.last_ac = None
        self.last_vac = None
        self.saved_weights = {}
        self.batch_size = 128
        self.epochs_num = 20
        self.vector_input1 = keras.Input(shape=(32,), name="R30_input_1")

        self.hidden_left_0 = layers.Dense(128, name="hidden_left_0", activation="tanh")(self.vector_input1)
        self.hidden_left_0 = layers.Dense(128, name="hidden_left_1", activation="linear")(self.hidden_left_0)
        self.hidden_left_1_1 = layers.Dense(128, name="hidden_left_1_1", activation="tanh")(self.hidden_left_0)
        self.hidden_left_2 = layers.Dense(128, name="hidden_left_2", activation="linear")(self.hidden_left_1_1)
        self.hidden_left_3 = layers.Dense(128, name="hidden_left_3", activation="tanh")(self.hidden_left_2)

        self.params_input1 = keras.Input(shape=(3,), name="module_params1")
        self.hidden_params1 = layers.Dense(64, name="hidden_params1",
                                           activation="tanh")(self.params_input1)

        self.params_input2 = keras.Input(shape=(3,), name="module_params2")
        self.hidden_params2 = layers.Dense(64, name="hidden_params2", activation="tanh")(self.params_input2)

        self.params_input3 = keras.Input(shape=(3,), name="module_params3")
        self.hidden_params3 = layers.Dense(64, name="hidden_params3",
                                           activation="tanh")(self.params_input3)

        self.params_input4 = keras.Input(shape=(3,), name="module_params4")
        self.hidden_params4 = layers.Dense(64, name="hidden_params4", activation="tanh")(self.params_input4)

        self.params_input5 = keras.Input(shape=(3,), name="module_params5")
        self.hidden_params5 = layers.Dense(64, name="hidden_params5",
                                           activation="tanh")(self.params_input5)

        self.params_input6 = keras.Input(shape=(3,), name="module_params6")
        self.hidden_params6 = layers.Dense(64, name="hidden_params6",
                                           activation="tanh")(self.params_input6)

        self.params_input7 = keras.Input(shape=(3,), name="module_params7")
        self.hidden_params7 = layers.Dense(64, name="hidden_params7",
                                           activation="tanh")(self.params_input7)

        self.params_input8 = keras.Input(shape=(3,), name="module_params8")
        self.hidden_params8 = layers.Dense(64, name="hidden_params8",
                                           activation="tanh")(self.params_input8)

        self.params_concatenate = layers.concatenate([self.hidden_params1, self.hidden_params2,
                                                      self.hidden_params3, self.hidden_params4,
                                                      self.hidden_params5, self.hidden_params6,
                                                      self.hidden_params7, self.hidden_params8
                                                      ])

        self.hidden_right_1 = layers.Dense(128, name="hidden_right_1", activation="linear")(self.params_concatenate)
        self.hidden_right_2 = layers.Dense(128, name="hidden_right_2", activation="tanh")(self.hidden_right_1)

        self.hidden_right_3 = layers.Dense(128, name="hidden_right_3", activation="linear")(self.hidden_right_2)
        self.hidden_right_4 = layers.Dense(128, name="hidden_right_4", activation="tanh")(self.hidden_right_3)
        self.hidden_right_5 = layers.Dense(128, name="hidden_right_5", activation="tanh")(self.hidden_right_4)

        self.middle_concatenate = layers.concatenate([self.hidden_left_3, self.hidden_right_5])

        self.hidden_middle1 = layers.Dense(128, name="hidden_middle1", activation="linear")(self.middle_concatenate)
        self.hidden_middle2 = layers.Dense(128, name="hidden_middle2", activation="tanh")(self.hidden_middle1)
        self.hidden_middle3 = layers.Dense(128, name="hidden_middle3", activation="relu")(self.hidden_middle2)
        self.hidden_middle4 = layers.Dense(128, name="hidden_middle4", activation="tanh")(self.hidden_middle3)
        self.hidden_middle41 = layers.Dense(128, name="hidden_middle41", activation="linear")(self.hidden_middle4)
        self.hidden_middle42 = layers.Dense(128, name="hidden_middle42", activation="tanh")(self.hidden_middle41)
        self.hidden_middle5 = layers.Dense(128, name="hidden_middle5", activation="linear")(self.hidden_middle42)

        self.output_layer = layers.Dense(32, name="output_layer")(self.hidden_middle5)
        self.model = keras.Model(
            inputs=[self.vector_input1, self.params_input1, self.params_input2,
                    self.params_input3, self.params_input4, self.params_input5,
                    self.params_input6, self.params_input7, self.params_input8],
            outputs=[self.output_layer],
        )
        opt = tf.optimizers.Adam(lr=1e-3, decay=1e-3 / 200)
        rms = tf.keras.metrics.RootMeanSquaredError()
        mse = tf.keras.losses.MeanSquaredError()
        self.model.compile(loss=mse, optimizer=opt, metrics=[rms])
        self.model.summary()
        keras.utils.plot_model(self.model, "my_model.png", show_shapes=True)

    def sub_pred(self, X):
        ret = []
        applied_modules = X[0]
        p_in_32 = X[1]
        ret.append(np.array(p_in_32).reshape((1, 32)))
        for i in range(0, 8):
            if i < len(applied_modules):
                mod = applied_modules[i]
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
            tmp = np.array([mod_id, param1, param2]).reshape((1, 3))
            ret.append(tmp)
        return ret, len(applied_modules)

    def predict_same_size_mod(self, X, mod_id):
        p = []
        mod = {}
        for i in range(1, 9):
            mod["mod{}".format(i)] = []
        for elm in X:
            p.append(elm[0])

            for i in range(1, 9):
                mod["mod{}".format(i)].append(elm[i])
        params = []
        p = tf.convert_to_tensor(np.array(p).reshape(len(X), 32))
        params.append(p)
        for i in range(1, 9):
            params.append(tf.convert_to_tensor(np.array(mod["mod{}".format(i)]).reshape(len(X), 3)))
        del mod
        # self.disable_layers(1, mod_id)
        # self.disable_layers(mod_id + 1, 8)
        ret = self.model.predict(params)
        # self.enable_layers()
        params = []
        for elm in ret:
            params.append(elm)
        return params

    def predict(self, X):
        if len(X) == 3:
            return self.model.predict(self.sub_pred(X)[0])
        if X is None:
            return None
        Xes = {}
        for i in range(1, 9):
            Xes[i] = []

        for i in range(1, len(X)):
            vec, mod_size = self.sub_pred(X[i])
            Xes[mod_size].append((i, vec))
        ret = np.empty((len(X), 32))
        print("Done ordering")
        for i in range(1, 9):
            if len(Xes[i]) > 0:
                Real_index, X_i = list(zip(*Xes[i]))
                tmp_res = self.predict_same_size_mod(X_i, i)
                assert (len(tmp_res) == len(Real_index))
                for j in range(0, len(Real_index)):
                    ret[Real_index[j]] = tmp_res[j]
        return ret

    def evaluate(self, X, Y):
        self.model.evaluate(X, Y)

    def fit(self, X_train, Y_train):
        X_P30 = []
        MOD1 = []
        MOD2 = []
        MOD3 = []
        MOD4 = []
        MOD5 = []
        MOD6 = []
        MOD7 = []
        MOD8 = []
        Y_P30 = []
        for i in range(0, len(X_train)):
            example_train = X_train[i]
            applied_modules = example_train[0]
            p_in_32 = example_train[1]
            if np.all(p_in_32 == 0):
                continue
            length = len(applied_modules)
            X_P30.append(np.array(p_in_32).reshape((1, 32)))
            Y_P30.append(np.array(Y_train[i]).reshape((1, 32)))
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
                tmp = np.array([mod_id, param1, param2]).reshape((1, 3))
                if k == 0:
                    MOD1.append(tmp)
                elif k == 1:
                    MOD2.append(tmp)
                elif k == 2:
                    MOD3.append(tmp)
                elif k == 3:
                    MOD4.append(tmp)
                elif k == 4:
                    MOD5.append(tmp)
                elif k == 5:
                    MOD6.append(tmp)
                elif k == 6:
                    MOD7.append(tmp)
                elif k == 7:
                    MOD8.append(tmp)

        total_len = 0
        print("Total len : {}".format(len(X_P30)))
        try:
            self.model.fit(
                x=
                {
                    "R30_input_1": np.array(X_P30).reshape(len(X_P30), 32),
                    "module_params1": np.array(MOD1).reshape((len(MOD1), 3)),
                    "module_params2": np.array(MOD2).reshape((len(MOD1), 3)),
                    "module_params3": np.array(MOD3).reshape((len(MOD1), 3)),
                    "module_params4": np.array(MOD4).reshape((len(MOD1), 3)),
                    "module_params5": np.array(MOD5).reshape((len(MOD1), 3)),
                    "module_params6": np.array(MOD6).reshape((len(MOD1), 3)),
                    "module_params7": np.array(MOD7).reshape((len(MOD1), 3)),
                    "module_params8": np.array(MOD8).reshape((len(MOD1), 3))
                },
                y=np.array(Y_P30).reshape((len(Y_P30), 32)),
                epochs=self.epochs_num, batch_size=self.epochs_num
            )
        except Exception as inst:
            print("exception : {}".format(inst))

