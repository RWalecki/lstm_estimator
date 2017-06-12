from sklearn.base import BaseEstimator
import tensorflow.contrib.keras as K
from tensorflow.contrib.keras import layers
from tensorflow.contrib.keras.python.keras.layers.wrappers import Bidirectional
import tensorflow as tf
import numpy as np
import shutil
import json
import os


from .utils import seq_corr, seq_rmse, data_generator



class lstm_estimator(BaseEstimator):
    def __init__(self, verbose=False, reg_0=0, reg_1=0, activation='relu', normalization=True, epochs=100, lr=1., dropout_1=0.5, dropout_2=0.5, dropout_3=0, dropout_4=0, units_1=256, units_2=256, log_dir='/tmp/LSTM',bidirectional=False, max_length=20):

        self.dropout_1 = dropout_1
        self.dropout_2 = dropout_2
        self.dropout_3 = dropout_3
        self.dropout_4 = dropout_4
        self.reg_0 = reg_0
        self.reg_1 = reg_1
        self.max_length = max_length
        self.units_1 = units_1
        self.units_2 = units_2
        self.bidirectional = bidirectional
        self.verbose = verbose
        self.normalization = normalization
        self.activation = activation
        self.epochs = epochs
        self.lr = lr
        self.log_dir = log_dir

        self.mode = 'regression'

        shutil.rmtree(log_dir, True)
        os.makedirs(log_dir)

        with open(log_dir+'/setting.json','w') as f:
            json.dump(self.__dict__, f, sort_keys=True)

    def _build_model(self, X, Y):

        self.input_shape = X[0].shape[1]
        self.output_shape = Y[0].shape[1:] 

        
        inputs = layers.Input(shape=[None, self.input_shape])


        if self.normalization:
            net = layers.BatchNormalization()(inputs)
        else:
            net = inputs

        if self.dropout_1:
            net = layers.Dropout(self.dropout_1)(net)

        if self.units_1:
            net = layers.Dense(self.units_1, activation=self.activation)(net)

        if self.normalization:
            net = layers.BatchNormalization()(net)

        if self.units_2:
            lstm_cell = layers.LSTM(
                    self.units_2, 
                    dropout = self.dropout_3,
                    recurrent_dropout = self.dropout_4,
                    kernel_regularizer= K.regularizers.l2(self.reg_0),
                    recurrent_regularizer = K.regularizers.l2(self.reg_1),
                    return_sequences=True 
                    )
            if self.bidirectional:lstm_cell = Bidirectional(lstm_cell)
            net = lstm_cell(net)

        if self.dropout_2:
            net = layers.Dropout(self.dropout_2)(net)


        outputs = layers.Dense(self.output_shape[0])(net)

        
        model = K.models.Model(inputs=inputs, outputs=outputs)

        model.compile(
                optimizer = K.optimizers.Adadelta(
                    lr = self.lr,
                    rho = 0.95, 
                    epsilon = 1e-08, 
                    decay = 0,
                    ),
                metrics = [seq_corr(self.mode), seq_rmse(self.mode)],
                loss = 'mean_squared_error',
                )

        model.summary()

        return model

    def fit(self, X_tr, Y_tr, X_val, Y_val):
        self.model = self._build_model(X_tr,Y_tr)
        gen_tr = data_generator(X_tr, Y_tr,  max_length=self.max_length)
        gen_val= data_generator(X_val,Y_val)
        x,y = next(gen_tr)

        self.model.fit_generator(
                gen_tr, 
                steps_per_epoch = len(X_tr)*20, 
                epochs = self.epochs, 
                validation_data = gen_val, 
                validation_steps = len(X_val),
                callbacks=[
                    K.callbacks.ModelCheckpoint(self.log_dir+'/model_best.h5', monitor='val_corr', save_best_only=True, mode='max'),
                    K.callbacks.ModelCheckpoint(self.log_dir+'/model_last.h5', monitor='val_corr', save_best_only=False),
                    K.callbacks.TensorBoard(log_dir=self.log_dir),
                    K.callbacks.CSVLogger(self.log_dir+'/summary.csv', separator=',', append=False)
                    ],
                verbose=self.verbose
                )

    def predict(self, X, best=True):
        if best:self.model.load_weights(self.log_dir+'/model_best.h5')
        out = []
        for x in X:
            out.append(self.model.predict(x[None,::])[0])
        return out
