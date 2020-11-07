import time

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import problem


def create_chunks(list_name, n):
    for i in range(0, len(list_name), n):
        yield list_name[i:i + n]


class SubModel:
    def __init__(self):
        self.last_ac = None
        self.saved_weights = {}
        self.batch_size = 32
        self.epochs_num = 10000
        self.vector_input1 = keras.Input(shape=(32,), name="R30_input_1")

        self.params_input1 = keras.Input(shape=(3,), name="module_params1")
        self.hidden_params1 = layers.Dense(32, name="hidden_params1", activation="relu")(self.params_input1)
        self.general_input1 = layers.concatenate([self.vector_input1, self.hidden_params1])
        self.hidden_layer_1_1 = layers.Dense(32, activation="tanh", name="hidden_layer_1_1")(self.general_input1)

        self.params_input2 = keras.Input(shape=(3,), name="module_params2")
        self.hidden_params2 = layers.Dense(32, name="hidden_params2", activation="relu")(self.params_input2)
        self.general_input2 = layers.concatenate([self.hidden_layer_1_1, self.hidden_params2])
        self.hidden_layer_2_1 = layers.Dense(32, activation="tanh", name="hidden_layer_2_1")(self.general_input2)

        self.params_input3 = keras.Input(shape=(3,), name="module_params3")
        self.hidden_params3 = layers.Dense(32, name="hidden_params3", activation="relu")(self.params_input3)
        self.general_input3 = layers.concatenate([self.hidden_layer_2_1, self.hidden_params3])
        self.hidden_layer_3_1 = layers.Dense(32, activation="tanh", name="hidden_layer_3_1")(self.general_input3)

        self.params_input4 = keras.Input(shape=(3,), name="module_params4")
        self.hidden_params4 = layers.Dense(32, name="hidden_params4", activation="relu")(self.params_input4)
        self.general_input4 = layers.concatenate([self.hidden_layer_3_1, self.hidden_params4])
        self.hidden_layer_4_1 = layers.Dense(32, activation="tanh", name="hidden_layer_4_1")(self.general_input4)

        self.params_input5 = keras.Input(shape=(3,), name="module_params5")
        self.hidden_params5 = layers.Dense(32, name="hidden_params5", activation="relu")(self.params_input5)
        self.general_input5 = layers.concatenate([self.hidden_layer_4_1, self.hidden_params5])
        self.hidden_layer_5_1 = layers.Dense(32, activation="tanh", name="hidden_layer_5_1")(self.general_input5)

        self.params_input6 = keras.Input(shape=(3,), name="module_params6")
        self.hidden_params5 = layers.Dense(32, name="hidden_params6", activation="relu")(self.params_input6)
        self.general_input6 = layers.concatenate([self.hidden_layer_5_1, self.hidden_params5])
        self.hidden_layer_6_1 = layers.Dense(32, activation="tanh", name="hidden_layer_6_1")(self.general_input6)

        self.params_input7 = keras.Input(shape=(3,), name="module_params7")
        self.hidden_params7 = layers.Dense(32, name="hidden_params7", activation="relu")(self.params_input7)
        self.general_input7 = layers.concatenate([self.hidden_layer_6_1, self.hidden_params7])
        self.hidden_layer_7_1 = layers.Dense(32, activation="tanh", name="hidden_layer_7_1")(self.general_input7)

        self.params_input8 = keras.Input(shape=(3,), name="module_params8")
        self.hidden_params8 = layers.Dense(32, name="hidden_params8", activation="relu")(self.params_input8)
        self.general_input8 = layers.concatenate([self.hidden_layer_7_1, self.hidden_params8])
        self.hidden_layer_8_1 = layers.Dense(32, activation="tanh", name="hidden_layer_8_1")(self.general_input8)

        self.model = keras.Model(
            inputs=[self.vector_input1, self.params_input1, self.params_input2,
                    self.params_input3, self.params_input4, self.params_input5,
                    self.params_input6, self.params_input7, self.params_input8],
            outputs=[self.hidden_layer_8_1],
        )

        self.model.compile()
        self.model.summary()
        keras.utils.plot_model(self.model, "my_model.png", show_shapes=True)

    # Original X_train, Y_train are passed directly here
    def enable_layers(self):
        for layer_name in self.saved_weights:
            self.model.get_layer(layer_name).Trainable = True
            self.model.get_layer(layer_name).set_weights(self.saved_weights[layer_name])
        self.saved_weights = {}

    def disable_layers(self, start, stop=9):
        for layer in range(start, stop):
            # print("disable from {}/8".format(layer))

            # input_param_layer_name = "module_params{}".format(layer + 1)
            hidden_param_layer_name = "hidden_params{}".format(layer)

            self.model.get_layer(hidden_param_layer_name).Trainable = False
            tmp = self.model.get_layer(hidden_param_layer_name).get_weights()
            self.saved_weights[hidden_param_layer_name] = tmp
            shape1 = tmp[0].shape
            shape2 = tmp[1].shape
            self.model.get_layer(hidden_param_layer_name).set_weights([np.zeros(shape1), np.zeros(shape2)])

    def fit(self, X_train, Y_train):
        # X_trains is splitted into an array called X_train_same_mod_size (delete all elms where R32 = np.zeros(32))
        # The first element of this array contains all X_train element that has one module applied
        # The second element of this array contains all X_train element that has two modules applied etc
        # Same goes with Y_train
        X_P30 = [[], [], [], [], [], [], [], []]
        MOD1 = [[], [], [], [], [], [], [], []]
        MOD2 = [[], [], [], [], [], [], [], []]
        MOD3 = [[], [], [], [], [], [], [], []]
        MOD4 = [[], [], [], [], [], [], [], []]
        MOD5 = [[], [], [], [], [], [], [], []]
        MOD6 = [[], [], [], [], [], [], [], []]
        MOD7 = [[], [], [], [], [], [], [], []]
        MOD8 = [[], [], [], [], [], [], [], []]

        Y_P30 = [[], [], [], [], [], [], [], []]

        for i in range(0, len(X_train)):
            example_train = X_train[i]
            applied_modules = example_train[0]
            p_in_32 = example_train[1]
            if np.all(p_in_32 == 0):
                continue

            length = len(applied_modules)
            X_P30[length - 1].append(np.array(p_in_32).reshape((1, 32)))
            Y_P30[length - 1].append(np.array(Y_train[i]).reshape((1, 32)))

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
                    MOD1[length - 1].append(tmp)
                elif k == 1:
                    MOD2[length - 1].append(tmp)
                elif k == 2:
                    MOD3[length - 1].append(tmp)
                elif k == 3:
                    MOD4[length - 1].append(tmp)
                elif k == 4:
                    MOD5[length - 1].append(tmp)
                elif k == 5:
                    MOD6[length - 1].append(tmp)
                elif k == 6:
                    MOD7[length - 1].append(tmp)
                elif k == 7:
                    MOD8[length - 1].append(tmp)

        total_len = 0
        for i in range(0, 8):
            total_len += len(X_P30[i])
        print("Total len : {}".format(total_len))
        train_dataset_mod = []
        for i in range(0, 8):
            tmp = tf.data.Dataset.from_tensor_slices((X_P30[i], MOD1[i],
                                                      MOD2[i], MOD3[i],
                                                      MOD4[i], MOD5[i],
                                                      MOD6[i], MOD7[i],
                                                      MOD8[i], Y_P30[i]
                                                      ))
            tmp = tmp.shuffle(buffer_size=1024).batch(self.batch_size)
            train_dataset_mod.append(tmp)

        train_acc_metric = keras.metrics.RootMeanSquaredError()
        val_acc_metric = keras.metrics.RootMeanSquaredError()
        loss_fn = keras.losses.MeanSquaredError()
        optimizer = keras.optimizers.Adam(1e-3, decay=1e-3 / 200)
        # Pre-trainning

        # Trainning
        for epoch in range(0, self.epochs_num):
            print("\nEpoch({})".format(epoch))
            start_time = time.time()
            for mods_num in range(0, 8):
                train_dataset = train_dataset_mod[mods_num]
                self.disable_layers(mods_num + 2)
                for step, (xp30_bt, mod1, mod2, mod3, mod4, mod5, mod6, mod7, mod8, yp30_bt) in enumerate(
                        train_dataset):
                    with tf.GradientTape() as tape:
                        logits = self.model([xp30_bt, mod1, mod2, mod3, mod4, mod5, mod6, mod7, mod8], training=True)

                        loss_value = loss_fn(yp30_bt, logits)
                    grads = tape.gradient(loss_value, self.model.trainable_weights)
                    optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
                    train_acc_metric.update_state(yp30_bt, logits)
                    if step % 200 == 0:
                        print("Sample : {} -- loss : {}".format((step + 1) * self.batch_size, float(loss_value)))

                self.enable_layers()

            train_acc = train_acc_metric.result()
            if self.last_ac:
                print("Time : {} -- Acc : {} -- Last diff : {}".format(time.time() - start_time, float(train_acc),
                                                                       float(self.last_ac) - float(train_acc)))
            else:
                print("Time : {} -- Acc : {} -- Last diff : None".format(time.time() - start_time, float(train_acc)))
            self.last_ac = float(train_acc)
            train_acc_metric.reset_states()

    def predict(self, X):
        # R32 + (M1 + M1P1 + M1P2) + ... + (M8 + M8P1 + M8P2)
        return self.model.predict(X)


a = SubModel()

X30 = np.ones((1, 32), dtype=float)
X_param = np.ones((1, 3), dtype=float)

Y = np.ones((1, 32), dtype=float)
# x = [X30, X_param, X_param, X_param, X_param, X_param, X_param, X_param, X_param]
# w =[x, x]
# with tf.GradientTape() as tape:
#    z = a.model(w, training=True)

# print(z.shape)

X_train, y_train = problem.get_train_data()
X_test, y_test = problem.get_test_data()

a.fit(X_train, y_train)