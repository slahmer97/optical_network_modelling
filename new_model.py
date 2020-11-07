import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import problem


class SubModel:
    def __init__(self):
        self.batch_size = 32
        self.vector_input1 = keras.Input(shape=(32,), name="R30_input_1")

        self.params_input1 = keras.Input(shape=(3,), name="module_params1")
        self.general_input1 = layers.concatenate([self.vector_input1, self.params_input1])

        self.hidden_layer_1_1 = layers.Dense(32, activation="tanh", name="hidden_layer_1_1")(self.general_input1)

        self.params_input2 = keras.Input(shape=(3,), name="module_params2")
        self.general_input2 = layers.concatenate([self.hidden_layer_1_1, self.params_input2])
        self.hidden_layer_2_1 = layers.Dense(32, activation="tanh", name="hidden_layer_2_1")(self.general_input2)

        self.params_input3 = keras.Input(shape=(3,), name="module_params3")
        self.general_input3 = layers.concatenate([self.hidden_layer_2_1, self.params_input3])
        self.hidden_layer_3_1 = layers.Dense(32, activation="tanh", name="hidden_layer_3_1")(self.general_input3)

        self.params_input4 = keras.Input(shape=(3,), name="module_params4")
        self.general_input4 = layers.concatenate([self.hidden_layer_3_1, self.params_input4])
        self.hidden_layer_4_1 = layers.Dense(32, activation="tanh", name="hidden_layer_4_1")(self.general_input4)

        self.params_input5 = keras.Input(shape=(3,), name="module_params5")
        self.general_input5 = layers.concatenate([self.hidden_layer_4_1, self.params_input5])
        self.hidden_layer_5_1 = layers.Dense(32, activation="tanh", name="hidden_layer_5_1")(self.general_input5)

        self.params_input6 = keras.Input(shape=(3,), name="module_params6")
        self.general_input6 = layers.concatenate([self.hidden_layer_5_1, self.params_input6])
        self.hidden_layer_6_1 = layers.Dense(32, activation="tanh", name="hidden_layer_6_1")(self.general_input6)

        self.params_input7 = keras.Input(shape=(3,), name="module_params7")
        self.general_input7 = layers.concatenate([self.hidden_layer_6_1, self.params_input7])
        self.hidden_layer_7_1 = layers.Dense(32, activation="tanh", name="hidden_layer_7_1")(self.general_input7)

        self.params_input8 = keras.Input(shape=(3,), name="module_params8")
        self.general_input8 = layers.concatenate([self.hidden_layer_7_1, self.params_input8])
        self.hidden_layer_8_1 = layers.Dense(32, activation="tanh", name="hidden_layer_8_1")(self.general_input8)

        self.model = keras.Model(
            inputs=[self.vector_input1, self.params_input1, self.params_input2,
                    self.params_input3, self.params_input4, self.params_input5,
                    self.params_input6, self.params_input7, self.params_input8],
            outputs=[self.hidden_layer_8_1],
        )

        self.model.compile(
            optimizer=keras.optimizers.RMSprop(1e-3),
            loss=[
                keras.losses.MeanSquaredError()
            ],
        )
        self.model.summary()
        keras.utils.plot_model(self.model, "my_model.png", show_shapes=True)

    # Original X_train, Y_train are passed directly here
    def fit(self, X_train, Y_train):
        # X_trains is splitted into an array called X_train_same_mod_size (delete all elms where R32 = np.zeros(32))
        # The first element of this array contains all X_train element that has one module applied
        # The second element of this array contains all X_train element that has two modules applied etc
        # Same goes with Y_train
        X_new_train = [[], [], [], [], [], [], [], []]
        Y_new_train = [[], [], [], [], [], [], [], []]

        for i in range(0, len(X_train)):
            example_train = X_train[i]
            applied_modules = example_train[0]
            p_in_30 = example_train[1]
            if np.all(p_in_30 == 0):
                continue

            length = len(applied_modules)
            tmp = [np.array(p_in_30)]
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
                tmp.append(np.array([mod_id, param1, param2]))

            X_new_train[length - 1].append(tmp)
            Y_new_train[length - 1].append(Y_train[i])

        print("len train : {}".format(len(X_train)))
        total_len = 0
        for i in range(0, 8):
            total_len += len(X_new_train[i])
        print("new len : {}".format(total_len))
        # Create X_train_batches array: first element of this array is the result of splitting of the first element
        # X_train_same_mod_size into equally size batches
        # same goes for all elements

        X_Y_multiset_batches = []
        for i in range(0, 8):
            X_new_train_i = X_new_train[i]
            Y_new_train_i = Y_new_train[i]
            x_y_ith_zipped = np.array(list(zip(X_new_train_i, Y_new_train_i)))
            np.random.shuffle(x_y_ith_zipped)
            X_new_train_i, Y_new_train_i = zip(*x_y_ith_zipped)
            X_new_train_i = list(X_new_train_i)
            Y_new_train_i = list(Y_new_train_i)
            batches_num = int(len(X_new_train_i) / self.batch_size)
            X_training_data = np.array_split(X_new_train_i, batches_num)
            Y_training_data = np.array_split(Y_new_train_i, batches_num)
            X_Y_multiset_batches.append((X_training_data, Y_training_data))

        for i in range(0, 8):
            (X_mod_i_batches, Y_mod_i_batches) = X_Y_multiset_batches[i]
            batches_num = len(X_mod_i_batches)
            for k in range(0, batches_num):
                X_mod_batch_i_k = X_mod_i_batches[k]
                Y_mod_batch_i_k = Y_mod_i_batches[k]
                print("X_mod_batch_i_k len : {}".format(len(X_mod_batch_i_k)))
                print("Y_mod_batch_i_k len : {}".format(len(Y_mod_batch_i_k)))
                return
        # For epoch in range(epoches_num):
        #   For X_size_batches in X_train_batches:
        #       For x_batch,y_batch in enumerate(X_size_batches) :
        #           at this point we got a batch of data where all element have the same number of modules Z applied
        #           disable the layers module_paramZ..8 (from Z to 8) so no weight update will be done at this point
        #           for each layer get_weights, save_them, zero_them
        #
        #           apply forward+back-propagation
        #
        #       restore saved weights, enable disactivated layers

    # vector of 56 element
    def predict(self, X):
        # R32 + (M1 + M1P1 + M1P2) + ... + (M8 + M8P1 + M8P2)
        return self.model.predict(X)


a = SubModel()

arr = np.ones(32 + 3 * 8, dtype=float)
a.model.get_layer("hidden_layer_8_1").trainable = False

X30 = np.ones((1, 32), dtype=float)
X_param = np.ones((1, 3), dtype=float)

Y = np.ones((1, 32), dtype=float)
"""
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = np.reshape(x_train, (-1, 784))
x_test = np.reshape(x_test, (-1, 784))
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(10)
"""
# z = a.predict([X30, X_param, X_param, X_param, X_param, X_param, X_param, X_param, X_param])
# print(z.shape)


X_train, y_train = problem.get_train_data()
X_test, y_test = problem.get_test_data()

a.fit(X_train, y_train)
