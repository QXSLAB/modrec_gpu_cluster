import numpy as np
from sklearn.datasets import make_classification
from torch import nn
import torch.nn.functional as F
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from skorch import NeuralNetClassifier
from skorch.callbacks import Checkpoint, EarlyStopping
import copy


X, y = make_classification(1000, 20, n_informative=10, random_state=0)
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


class SaveBestParam(Checkpoint):

    """Save best model state"""

    def save_model(self, net):
        self.best_model_dict = copy.deepcopy(
            net.module_.state_dict()
        )


class StopRestore(EarlyStopping):

    """Early Stop and Restore best module state"""

    def on_epoch_end(self, net, **kwargs):
        # super().on_epoch_end(net, **kwargs)
        current_score = net.history[-1, self.monitor]
        if not self._is_score_improved(current_score):
            self.misses_ += 1
        else:
            self.misses_ = 0
            self.dynamic_threshold_ = self._calc_new_threshold(current_score)
        if self.misses_ == self.patience:
            best_cp = net.get_params()['callbacks__best']
            net.module_.load_state_dict(best_cp.best_model_dict)
            if net.verbose:
                self._sink("Stopping since {} has not improved in the last "
                           "{} epochs.".format(self.monitor, self.patience),
                           verbose=net.verbose)
            raise KeyboardInterrupt


cp = SaveBestParam(dirname='best')
early_stop = StopRestore(patience=20)
net = NeuralNetClassifier(
    MyModule,
    max_epochs=1000,
    callbacks=[('best', cp),
               ('early', early_stop)],
    lr=0.1,
)


param_dist = {
    'lr': [0.51, 0.52, 0.53, 0.54],
}

search = RandomizedSearchCV(net,
                            param_dist,
                            cv=StratifiedKFold(n_splits=3),
                            n_iter=4,
                            verbose=10,
                            scoring='accuracy')

search.fit(X, y)
print('\n')
