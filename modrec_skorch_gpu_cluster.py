from skorch import NeuralNetClassifier
from skorch.callbacks import Checkpoint, EarlyStopping
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from scipy.stats import norm
from dask.distributed import Client
from sklearn.externals import joblib
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import pickle
#
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'


warnings.filterwarnings("ignore")

mods = ['BPSK', 'DQPSK', 'GFSK', 'GMSK', 'OQPSK',
        'PAM4', 'PAM8', 'PSK8', 'QAM16', 'QAM64', 'QPSK']
class_num = len(mods)


def import_from_mat(data, size):
    features = []
    labels = []
    for mod in mods:
        real = np.array(data[mod].real[:size])
        imag = np.array(data[mod].imag[:size])
        signal = np.concatenate([real, imag], axis=1)
        features.append(signal)
        labels.append(mods.index(mod) * np.ones([size, 1]))

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    return features, labels


def load_data():

    print("loading data")

    data = scipy.io.loadmat(
        "D:/batch100000_symbols128_sps8_baud1_snr5.dat",
    )
    features, labels = import_from_mat(data, 100000)
    features = features.astype(np.float32)
    labels = labels.astype(np.int64)
    X = features
    y = labels.reshape(-1)

    # class_num = 10
    # X, y = make_classification(100, 2048,
    #                            n_informative=5,
    #                            n_classes=class_num,
    #                            random_state=0)
    # X = X.astype(np.float32)
    # y = y.astype(np.int64)

    return X, y


class Discriminator(nn.Module):

    print("Define the model")

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


def train():

    disc = Discriminator()

    cp = Checkpoint(dirname='best')
    early_stop = EarlyStopping(patience=20)
    net = NeuralNetClassifier(
        disc,
        max_epochs=1000,
        lr=0.01,
        device='cuda',
        callbacks=[('best', cp),
                   ('early', early_stop)],
        iterator_train__shuffle=True,
        iterator_valid__shuffle=False
    )
    # net.set_params(callbacks__print_log=None)

    param_dist = {
        'lr': [0.001, 0.005, 0.01, 0.05],
    }

    search = RandomizedSearchCV(net,
                                param_dist,
                                cv=StratifiedKFold(n_splits=3),
                                n_iter=4,
                                verbose=10,
                                scoring='accuracy')

    X, y = load_data()

    # search.fit(X, y)

    client = Client("127.0.0.1:8786")  # create local cluster

    with joblib.parallel_backend('dask'):
        search.fit(X, y)

    with open('result.pkl', 'wb') as f:
        pickle.dump(search, f)


if __name__ == "__main__":
    train()
