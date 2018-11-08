import numpy as np
from dask.distributed import Client

from sklearn.externals import joblib
from sklearn.datasets import load_digits
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC

import warnings
warnings.filterwarnings("ignore")

client = Client("127.0.0.1:8786")             # create local cluster

# digits = load_digits()
#
# param_space = {
#     'C': np.logspace(-6, 6, 13),
#     'gamma': np.logspace(-8, 8, 17),
#     'tol': np.logspace(-4, -1, 4),
#     'class_weight': [None, 'balanced'],
# }
#
# model = SVC(kernel='rbf')
# search = RandomizedSearchCV(model, param_space, cv=3, n_iter=50, verbose=10)


import numpy as np
from sklearn.datasets import make_classification
from torch import nn
import torch.nn.functional as F

from skorch import NeuralNetClassifier


# X, y = make_classification(10000, 20, n_informative=10, random_state=0)
# X = X.astype(np.float32)
# y = y.astype(np.int64)
#
# class MyModule(nn.Module):
#     def __init__(self, num_units=10, nonlin=F.relu):
#         super(MyModule, self).__init__()
#
#         self.dense0 = nn.Linear(20, num_units)
#         self.nonlin = nonlin
#         self.dropout = nn.Dropout(0.5)
#         self.dense1 = nn.Linear(num_units, 10)
#         self.output = nn.Linear(10, 2)
#
#     def forward(self, X, **kwargs):
#         X = self.nonlin(self.dense0(X))
#         X = self.dropout(X)
#         X = F.relu(self.dense1(X))
#         X = F.softmax(self.output(X))
#         return X
#
#
# md = MyModule()
#
# net = NeuralNetClassifier(
#     md,
#     max_epochs=10,
#     lr=0.1,
# )
#
# net.set_params(callbacks__print_log=None)
#
# param_space = {
#     'lr': [0.1, 0.2, 0.2]
# }
#
# search = RandomizedSearchCV(net, param_space, cv=3, n_iter=50, verbose=10, scoring='accuracy')
#




import os
import numpy as np
import scipy.io

# mods = ['BPSK', 'DQPSK', 'GFSK', 'GMSK', 'OQPSK',
#         'PAM4', 'PAM8', 'PSK8', 'QAM16', 'QAM64', 'QPSK']
# class_num = len(mods)
#
# data = scipy.io.loadmat(
#     "D:/batch100000_symbols128_sps8_baud1_snr5.dat",
# )
#
#
# def import_from_mat(data, size):
#     features = []
#     labels = []
#     for mod in mods:
#         real = np.array(data[mod].real[:size])
#         imag = np.array(data[mod].imag[:size])
#         signal = np.concatenate([real, imag], axis=1)
#         features.append(signal)
#         labels.append(mods.index(mod) * np.ones([size, 1]))
#
#     features = np.concatenate(features, axis=0)
#     labels = np.concatenate(labels, axis=0)
#
#     return features, labels
#
# features, labels = import_from_mat(data,10)
#
#
# features = features.astype(np.float32)
# labels = labels.astype(np.int64)
#
# X = features
# y = labels.reshape(-1)

class_num = 10
X, y = make_classification(20, 2048, n_informative=class_num, random_state=0)
X = X.astype(np.float32)
y = y.astype(np.int64)
import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    """Define the model"""

    def __init__(self, dr=0.6):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(2, 256, 3, padding=1),  # batch, 256, 1024
            nn.BatchNorm1d(256),
            nn.ReLU(),
            # nn.Dropout2d()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(256, 80, 3, padding=1),  # batch, 80, 1024
            nn.BatchNorm1d(80),
            nn.ReLU(),
            # nn.Dropout2d()
        )
        self.fc1 = nn.Sequential(
            nn.Linear(80 * 1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=dr)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256, class_num),
            nn.ReLU()
        )

    def forward(self, x, **kwargs):
        x = x.reshape((x.size(0), 2, -1))
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x


from skorch import NeuralNetClassifier
from skorch.callbacks import Callback, EpochScoring, Checkpoint, EarlyStopping, PrintLog
from sklearn.metrics import confusion_matrix
from skorch.utils import data_from_dataset

d = Discriminator()

net = NeuralNetClassifier(
    d,
    max_epochs=200,
    lr=0.01,
    device='cuda',
    iterator_train__shuffle=True,
    iterator_valid__shuffle=False
)
net.set_params(callbacks__print_log=None)

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import norm

param_dist = {
    'lr': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05],
}

search = RandomizedSearchCV(net,
                            param_dist,
                            cv=3,
                            n_iter=50,
                            verbose=10,
                            scoring='accuracy')

with joblib.parallel_backend('dask'):
    search.fit(X, y)