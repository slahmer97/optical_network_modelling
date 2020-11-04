from torch.optim import optimizer

import problem
import numpy as np
import torch
import torch.nn as nn

X_train, y_train = problem.get_train_data()
X_test, y_test = problem.get_test_data()


class Regressor():

    def __init__(self, input_size=56, output_size=32):
        super().__init__()
        self.tmp = 0
        self.network = nn.Sequential(
            nn.Linear(input_size, input_size * 2),
            nn.Softmax(),
            nn.Linear(input_size * 2, output_size),
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
        X = torch.from_numpy(X).float().reshape(X_.shape[0], 56)
        Y = torch.from_numpy(Y_).float()

        loss_fn = torch.nn.MSELoss(reduction='sum')

        learning_rate = 1e-4
        opt = torch.optim.Adam(self.network.parameters(), lr=learning_rate)

        for t in range(100000):

            y_pred = self.network(X)

            loss = loss_fn(y_pred, Y)
            if t % 100 == 0:
                print(t, loss.item())

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
z.load()
z.fit(X_=X_train, Y_=y_train)
z.save()
res = z.predict(X_test[0])

print(res)
