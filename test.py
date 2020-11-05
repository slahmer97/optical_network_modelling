from torch.optim import optimizer

import problem
import numpy as np
import torch
import torch.nn as nn

X_train, y_train = problem.get_train_data()
X_test, y_test = problem.get_test_data()
"""
nn.Linear(input_size, input_size * 2),
nn.Softmax(),
nn.Linear(input_size * 2, output_size),
nn.ReLU()
"""


class Regressor():

    def __init__(self, input_size=56, output_size=32):
        super().__init__()
        self.tmp = 0
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LayerNorm(normalized_shape=256),
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5),
            nn.MaxPool1d(kernel_size=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5),
            nn.MaxPool1d(kernel_size=2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(25, 128),
            nn.Tanh(),
            nn.Linear(128, 32),
            nn.ReLU()
        )

    @staticmethod
    def process_input(x_):
        input_ = x_[0]
        ret_vec = np.empty([24], dtype=np.float64)
        i = 0
        for elm in input_:
            if elm[0] == 'EDFA':
                ret_vec[i] = 1
            if elm[0] == 'SMF':
                ret_vec[i] = -1
            ret_vec[i + 1] = elm[1][0]
            ret_vec[i + 2] = elm[1][1]
            i += 3

        for j in range(i, 24):
            ret_vec[j] = 0
        return np.concatenate((x_[1], ret_vec), axis=0)

    def fit(self, X_, Y_):
        self.tmp += 1
        X = np.empty(shape=(X_.shape[0], 56), dtype=np.float64)
        i = 0
        for training_example in X_:
            X[i] = Regressor.process_input(training_example)
            i += 1
        X = torch.from_numpy(X).float()
        print("before X shape : {}".format(X.size()))
        X = X.view(X.shape[0], 1, -1)
        print("X shape : {}".format(X.size()))
        Y = torch.from_numpy(Y_).float()
        Y = Y.view(Y.shape[0], 1, -1)
        loss_fn = torch.nn.MSELoss(reduction='sum')

        learning_rate = 1e-4
        opt = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        last = None
        for t in range(100000):

            y_pred = self.network(X)

            loss = loss_fn(y_pred, Y)
            if t % 100 == 0:
                if last:
                    diff = last - loss.item()
                    print("iter : {} -- loss : {} -- diff : {}".format(t, loss.item(), diff))

                last = loss.item()
            if t % 10000 == 0:
                self.save()
            opt.zero_grad()
            loss.backward()
            opt.step()

    def save(self):
        torch.save(self.network.state_dict(), "model")

    def load(self):
        self.network.load_state_dict(torch.load("model"))
        self.network.eval()

    def predict(self, X_):
        self.tmp += 1
        X = torch.from_numpy(Regressor.process_input(X_)).reshape((1, 56)).float()
        # print("X : {}".format(X))
        # print(X.shape)
        return self.network.forward(X)
        # return self.network.forward(X)


z = Regressor()
z.fit(X_=X_train, Y_=y_train)
z.save()
# res = z.predict(X_test[0])

# print(res)
