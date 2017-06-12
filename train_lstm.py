import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from sklearn.model_selection import ParameterGrid
from lstm_estimator import lstm_estimator
import numpy as np
np.random.seed(0)

######################################################################## 
# make some synthetic data: N sequences of lenght 1000 and 128 featrues
######################################################################## 
X_tr = [np.random.rand(1000,128) for i in range(5)]
X_te = [np.random.rand(1000,128) for i in range(3)]
Y_tr = [np.random.randint(0,2,[1000,12,6]) for i in range(5)]
Y_te = [np.random.randint(0,2,[1000,12,6]) for i in range(3)]
######################################################################## 

exp_id = int(sys.argv[1])
pwd = os.path.dirname(os.path.realpath(__file__))
log_dir = pwd+'/out/'+str(exp_id)


grid = list(ParameterGrid({
        'units_1':[128, 256],
        'units_2':[128, 256],
        'normalization':[True],
        'dropout_1':[0, 0.5],
        'dropout_2':[0, 0.5],
        'dropout_3':[0, 0.5],
        'dropout_4':[0, 0.5],
        'reg_0':[0]+[10.**i for i in range(-5,5)],
        'reg_1':[0]+[10.**i for i in range(-5,5)],
        'bidirectional':[True],
        'activation':['relu'],
        'lr':[1.,10.],
        'epochs':[1],
        }))

np.random.shuffle(grid)
print(log_dir)

clf = lstm_estimator(**grid[exp_id], verbose=1, log_dir=log_dir)

clf.fit(X_tr, Y_tr, X_te, Y_te)
Y_hat = clf.predict(X_te)

for seq_te, seq_hat in zip(Y_te, Y_hat):
    assert(seq_te.argmax(-1).shape == seq_hat.shape)
