import numpy as np
from dask.distributed import Client

from sklearn.externals import joblib
from sklearn.datasets import load_digits
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC

import warnings
warnings.filterwarnings("ignore")

client = Client(processes=False)             # create local cluster

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


X, y = make_classification(10000, 20, n_informative=10, random_state=0)
X = X.astype(np.float32)
y = y.astype(np.int64)

class MyModule(nn.Module):
    def __init__(self, num_units=10, nonlin=F.relu):
        super(MyModule, self).__init__()

        self.dense0 = nn.Linear(20, num_units)
        self.nonlin = nonlin
        self.dropout = nn.Dropout(0.5)
        self.dense1 = nn.Linear(num_units, 10)
        self.output = nn.Linear(10, 2)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        X = self.dropout(X)
        X = F.relu(self.dense1(X))
        X = F.softmax(self.output(X))
        return X


md = MyModule()

net = NeuralNetClassifier(
    md,
    max_epochs=10,
    lr=0.1,
)

net.set_params(callbacks__print_log=None)

param_space = {
    'lr': [0.1, 0.2, 0.2]
}

search = RandomizedSearchCV(net, param_space, cv=3, n_iter=50, verbose=10, scoring='accuracy')

with joblib.parallel_backend('dask'):
    search.fit(X, y)