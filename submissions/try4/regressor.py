import sklearn
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from sklearn.linear_model import LinearRegression


def process_data(X_train, Y_train=None, shifting=False):
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
        from collections import deque
        if shifting:
            Y_P30.append(np.array(Y_train[i]).reshape((1, 32)))
            new_tmp = deque()

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

            tmp = np.array([mod_id, param1 / 10.0, param2 / 10.0]).reshape((1, 3))
            if shifting:
                new_tmp.append(tmp)

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

        if shifting:
            for _ in range(length, 8):
                new_tmp.rotate(1)
                X_P30.append(np.array(p_in_32).reshape((1, 32)))
                Y_P30.append(np.array(Y_train[i]).reshape((1, 32)))

                for k in range(0, 8):
                    if k == 0:
                        MOD1.append(new_tmp[k])
                    elif k == 1:
                        MOD2.append(new_tmp[k])
                    elif k == 2:
                        MOD3.append(new_tmp[k])
                    elif k == 3:
                        MOD4.append(new_tmp[k])
                    elif k == 4:
                        MOD5.append(new_tmp[k])
                    elif k == 5:
                        MOD6.append(new_tmp[k])
                    elif k == 6:
                        MOD7.append(new_tmp[k])
                    elif k == 7:
                        MOD8.append(new_tmp[k])
    if shifting:
        assert len(X_P30) == len(MOD1) == len(MOD2) == len(MOD3) == len(MOD4) == len(MOD5) == len(MOD6) == len(Y_P30)
    return X_P30, MOD1, MOD2, MOD3, MOD4, MOD5, MOD6, MOD7, MOD8, Y_P30


class Regressor:
    def __init__(self, *args, **kwargs):
        if len(kwargs) == 0:
            return
        # import numpy as np
        id = kwargs.pop("id", None)
        print("Launching -- {} -- ".format(id))
        # np.random.seed(1337)
        self.last_ac = None
        self.last_vac = None
        self.epsilon = 0.0001
        self.saved_weights = {}
        self.batch_size = 64
        self.epochs_num = 100
        self.vector_input1 = keras.Input(shape=(32,), name="R30_input_1")

        # self.hidden_left_0 = layers.Dense(128, name="hidden_left_0", activation="linear")(self.vector_input1)
        # self.hidden_left_0 = layers.Dense(512, name="hidden_left_1", activation="tanh")(self.hidden_left_0)
        # self.hidden_left_1_1 = layers.Dense(512, name="hidden_left_1_1", activation="tanh")(self.hidden_left_0)
        # self.hidden_left_2 = layers.Dense(512, name="hidden_left_2", activation="tanh")(self.hidden_left_1_1)
        self.hidden_left_3 = layers.Dense(512, name="hidden_left_3", activation="tanh")(self.vector_input1)

        self.params_input1 = keras.Input(shape=(3,), name="module_params1")
        # self.hidden_params1_0 = layers.Dense(6, name="hidden_params1_0", activation="tanh")(self.params_input1)
        # self.hidden_params1 = layers.Dense(6, name="hidden_params1", activation="tanh")(self.hidden_params1_0)

        self.params_input2 = keras.Input(shape=(3,), name="module_params2")
        # self.hidden_params2_0 = layers.Dense(6, name="hidden_params2_0", activation="tanh")(self.params_input2)
        # self.hidden_params2 = layers.Dense(6, name="hidden_params2", activation="tanh")(self.hidden_params2_0)

        self.params_input3 = keras.Input(shape=(3,), name="module_params3")
        # self.hidden_params3_0 = layers.Dense(6, name="hidden_params3_0", activation="tanh")(self.params_input3)
        # self.hidden_params3 = layers.Dense(6, name="hidden_params3", activation="tanh")(self.hidden_params3_0)

        self.params_input4 = keras.Input(shape=(3,), name="module_params4")
        # self.hidden_params4_0 = layers.Dense(6, name="hidden_params4_0", activation="tanh")(self.params_input4)
        # self.hidden_params4 = layers.Dense(6, name="hidden_params4", activation="tanh")(self.hidden_params4_0)

        self.params_input5 = keras.Input(shape=(3,), name="module_params5")
        # self.hidden_params5_0 = layers.Dense(6, name="hidden_params5_0", activation="tanh")(self.params_input5)
        # self.hidden_params5 = layers.Dense(6, name="hidden_params5", activation="tanh")(self.hidden_params5_0)

        self.params_input6 = keras.Input(shape=(3,), name="module_params6")
        # self.hidden_params6_0 = layers.Dense(6, name="hidden_params6_0", activation="tanh")(self.params_input6)
        # self.hidden_params6 = layers.Dense(6, name="hidden_params6", activation="tanh")(self.hidden_params6_0)

        self.params_input7 = keras.Input(shape=(3,), name="module_params7")
        # self.hidden_params7_0 = layers.Dense(6, name="hidden_params7_0", activation="tanh")(self.params_input7)
        # self.hidden_params7 = layers.Dense(6, name="hidden_params7", activation="tanh")(self.hidden_params7_0)

        self.params_input8 = keras.Input(shape=(3,), name="module_params8")
        # self.hidden_params8_0 = layers.Dense(6, name="hidden_params8_0", activation="tanh")(self.params_input8)
        # self.hidden_params8 = layers.Dense(6, name="hidden_params8", activation="tanh")(self.hidden_params8_0)

        self.params_concatenate = layers.concatenate([self.params_input1, self.params_input2,
                                                      self.params_input3, self.params_input4,
                                                      self.params_input5, self.params_input6,
                                                      self.params_input7, self.params_input8
                                                      ])

        # self.hidden_right_1 = layers.Dense(256, name="hidden_right_1", activation="linear")(self.params_concatenate)
        # self.hidden_right_2 = layers.Dense(256, name="hidden_right_2", activation="tanh")(self.hidden_right_1)
        # self.hidden_right_3 = layers.Dense(256, name="hidden_right_3", activation="tanh")(self.hidden_right_2)
        # self.hidden_right_4 = layers.Dense(128, name="hidden_right_4", activation="tanh")(self.hidden_right_3)
        self.hidden_right_5 = layers.Dense(256, name="hidden_right_5", activation="tanh")(self.params_concatenate)

        self.middle_concatenate = layers.concatenate([self.hidden_left_3, self.hidden_right_5])

        self.hidden_middle1 = layers.Dense(256, name="hidden_middle1", activation="linear")(self.middle_concatenate)
        # self.hidden_middle2 = layers.Dense(256, name="hidden_middle2", activation="tanh")(self.hidden_middle1)
        # self.hidden_middle3 = layers.Dense(512, name="hidden_middle3", activation="tanh")(self.hidden_middle2)
        self.hidden_middle4 = layers.Dense(256, name="hidden_middle4", activation="tanh")(self.hidden_middle1)
        self.hidden_middle41 = layers.Dense(256, name="hidden_middle41", activation="linear")(self.hidden_middle4)
        self.hidden_middle42 = layers.Dense(256, name="hidden_middle42", activation="tanh")(self.hidden_middle41)
        self.hidden_middle5 = layers.Dense(256, name="hidden_middle5", activation="linear")(self.hidden_middle42)

        self.output_layer = layers.Dense(32, name="output_layer", activation="relu")(self.hidden_middle5)
        self.model = keras.Model(
            inputs=[self.vector_input1, self.params_input1, self.params_input2,
                    self.params_input3, self.params_input4, self.params_input5,
                    self.params_input6, self.params_input7, self.params_input8],
            outputs=[self.output_layer],
        )
        opt = tf.optimizers.Adam(lr=0.0001)
        rms = tf.keras.metrics.RootMeanSquaredError()
        mse = tf.keras.losses.MeanSquaredError()
        self.model.compile(loss=mse, optimizer=opt, metrics=[rms])
        # self.model.summary()

    def predict2(self, X):
        X_P30, MOD1, MOD2, MOD3, MOD4, MOD5, MOD6, MOD7, MOD8, _ = process_data(X)

        return self.model.predict(
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
            }
        )

    def predict(self, X):
        res = []
        for i in range(10):
            res.append(self.models[i].predict2(X))

        return np.mean(res, axis=0)

    def fit(self, X, Y):
        self.models = {}
        for i in range(10):
            self.models[i] = Regressor(id=i)
            self.models[i].fit2(X, Y)

    def fit2(self, X_train, Y_train):
        assert len(X_train) == len(Y_train)
        scores = np.empty(len(X_train))
        for i in range(0, len(X_train)):
            p30_in = np.array(X_train[i][1])
            p30_in = (p30_in[p30_in > 0.0]).reshape(-1, 1)

            p30_out = np.array(Y_train[i])
            p30_out = (p30_out[p30_out > 0.0]).reshape(-1, 1)

            reg = LinearRegression().fit(p30_in, p30_out)
            reg_score = reg.score(p30_in, p30_out)
            if reg_score < self.epsilon:
                reg_score = 0.0
            num_of_mods = len(X_train[i][0])
            scores[i] = 0.8 * num_of_mods + 0.2 * reg_score
        scores = scores / scores.sum()
        sample_size = int(len(X_train) * 0.6)
        print("step 1 done : {} -- sample_size = {}".format(scores.sum(), sample_size))
        indexes = np.random.choice(range(len(X_train)), size=sample_size, replace=False, p=scores)

        X = np.array(X_train)[indexes]
        Y = np.array(Y_train)[indexes]
        X_train = None
        Y_train = None
        X_P30, MOD1, MOD2, MOD3, MOD4, MOD5, MOD6, MOD7, MOD8, Y_P30 = process_data(X, Y, shifting=False)
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
                y=np.array(Y).reshape((len(Y), 32)),
                epochs=self.epochs_num, batch_size=self.batch_size
            )
        except Exception as inst:
            print("exception : {}".format(inst))
