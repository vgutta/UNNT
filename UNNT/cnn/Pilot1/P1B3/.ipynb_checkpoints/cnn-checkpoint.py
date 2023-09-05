from __future__ import division, print_function

import logging
import os
import time

import numpy as np
import pandas as pd

from keras import backend as K
from keras import metrics
from sklearn.metrics import r2_score, mean_squared_error
from keras.models import Sequential
from keras.layers import Activation, BatchNormalization, Dense, Dropout, LocallyConnected1D, Conv1D, MaxPooling1D, Flatten, Conv2D, LocallyConnected2D
from keras.callbacks import Callback, ModelCheckpoint, ProgbarLogger

# For non-interactive plotting
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


#import p1b3 as benchmark
from . import p1b3_fda_drugs as benchmark
from . import user_data
import candle

DATA_PATH = os.path.join(os.path.dirname(__file__), '../../../../', 'data')
CNN_CONFIG = os.path.join(os.path.dirname(__file__), '../../../cnn_config.txt')

# define class CNN
class CNN:
    #constructor
    def __init__(self, args, default_data=True, epochs=10, batch_size=32, verbose=1, validation_split=0.2):
            
            self.default_data = default_data
            self.target_variable = args.target_variable
            self.epochs = epochs
            self.batch_size = batch_size
            self.verbose = verbose
            self.validation_split = validation_split
            
            self.model = None
            self.start_time = None
            self.end_time = None
            self.runtime = None
            self.predictions = None
            self.rmse = None
            self.r2_error = None
            self.ext = None

            gParameters = initialize_parameters()
            fh, sh = self.setup(gParameters)

            # Get default parameters for initialization and optimizer functions
            kerasDefaults = candle.keras_default_config()
            seed = gParameters['rng_seed']

            if self.default_data:
                self.train_default_data_model(gParameters, kerasDefaults, seed)
            else:
                self.train_custom_data_model(gParameters, kerasDefaults, seed)
            
            benchmark.logger.removeHandler(fh)
            benchmark.logger.removeHandler(sh)

    def train_default_data_model(self, gParameters, kerasDefaults, seed):
        # Build dataset loader object
        loader = benchmark.DataLoader(seed=seed, dtype=gParameters['data_type'],
                        val_split=gParameters['val_split'],
                        test_cell_split=gParameters['test_cell_split'],
                        cell_features=gParameters['cell_features'],
                        drug_features=gParameters['drug_features'],
                        feature_subsample=gParameters['feature_subsample'],
                        scaling=gParameters['scaling'],
                        scramble=gParameters['scramble'],
                        min_logconc=gParameters['min_logconc'],
                        max_logconc=gParameters['max_logconc'],
                        subsample=gParameters['subsample'],
                        category_cutoffs=gParameters['category_cutoffs'])

        # Initialize weights and learning rule
        initializer_weights = candle.build_initializer(gParameters['initialization'], kerasDefaults, seed)
        initializer_bias = candle.build_initializer('constant', kerasDefaults, 0.)

        model, gen_shape = self.define_model(gParameters, loader, initializer_weights, initializer_bias)

        # Define optimizer
        optimizer = candle.build_optimizer(gParameters['optimizer'],
                                        gParameters['learning_rate'],
                                        kerasDefaults)

        # Compile and display model
        model.compile(loss=gParameters['loss'], optimizer=optimizer)
        model.summary()
        benchmark.logger.debug('Model: {}'.format(model.to_json()))

        train_gen = benchmark.DataGenerator(loader, batch_size=gParameters['batch_size'], shape=gen_shape, cell_noise_sigma=gParameters['cell_noise_sigma']).flow()
        val_gen = benchmark.DataGenerator(loader, partition='val', batch_size=gParameters['batch_size'], shape=gen_shape).flow()
        val_gen2 = benchmark.DataGenerator(loader, partition='val', batch_size=gParameters['batch_size'], shape=gen_shape).flow()
        test_gen = benchmark.DataGenerator(loader, partition='test', batch_size=gParameters['batch_size'], shape=gen_shape).flow()

        train_steps = int(loader.n_train/gParameters['batch_size'])
        val_steps = int(loader.n_val/gParameters['batch_size'])
        test_steps = int(loader.n_test/gParameters['batch_size'])

        if 'train_steps' in gParameters:
            train_steps = gParameters['train_steps']
        if 'val_steps' in gParameters:
            val_steps = gParameters['val_steps']
        if 'test_steps' in gParameters:
            test_steps = gParameters['test_steps']

        checkpointer = ModelCheckpoint(filepath=gParameters['output_dir']+'.model'+self.ext+'.h5', save_best_only=True)
        progbar = MyProgbarLogger(train_steps * gParameters['batch_size'])
        loss_history = MyLossHistory(progbar=progbar, val_gen=val_gen2, test_gen=test_gen,
                                val_steps=val_steps, test_steps=test_steps,
                                metric=gParameters['loss'], category_cutoffs=gParameters['category_cutoffs'],
                                ext=self.ext, pre=gParameters['output_dir'])

        # Seed random generator for training
        np.random.seed(seed)

        candleRemoteMonitor = candle.CandleRemoteMonitor(params=gParameters)

        start_time = time.time()
        
        history = model.fit_generator(train_gen, train_steps,
                            epochs=gParameters['epochs'],
                            validation_data=val_gen,
                            validation_steps=val_steps,
                            verbose=0,
                            callbacks=[loss_history, progbar, candleRemoteMonitor],
                            )
        
        end_time = time.time()
        self.runtime = round(end_time - start_time, 2)

        print("Training Time: ", self.runtime, " seconds")

        
        # test set
        test_data = pd.read_csv(os.path.join(DATA_PATH, 'val_data_nci60.csv'), sep=',')
        test_data = test_data.dropna()

        test_y = test_data['AUC'].to_numpy()
        test_drugs = test_data['NSC']

        test_data = test_data.drop(['SOURCE', 'CELLNAME', 'NSC', 'AUC'], 1)
        test_data = test_data.to_numpy()

        y_pred = model.predict(test_data)
        self.r2_error = round(r2_score(test_y, y_pred), 4)

        mse = mean_squared_error(test_y, y_pred)
        self.rmse = round(np.sqrt(mse), 4)
        


    def train_custom_data_model(self, gParameters, kerasDefaults, seed):
        # Build dataset loader object
        loader = user_data.DataLoader(self.args)

        # Initialize weights and learning rule
        initializer_weights = candle.build_initializer(gParameters['initialization'], kerasDefaults, seed)
        initializer_bias = candle.build_initializer('constant', kerasDefaults, 0.)

        model, gen_shape = self.define_model(gParameters, loader, initializer_weights, initializer_bias)

        # Define optimizer
        optimizer = candle.build_optimizer(gParameters['optimizer'],
                                        gParameters['learning_rate'],
                                        kerasDefaults)

        # Compile and display model
        model.compile(loss=gParameters['loss'], optimizer=optimizer)
        model.summary()
        benchmark.logger.debug('Model: {}'.format(model.to_json()))

        train_gen = benchmark.DataGenerator(loader, batch_size=gParameters['batch_size'], shape=gen_shape).flow()
        val_gen = benchmark.DataGenerator(loader, partition='val', batch_size=gParameters['batch_size'], shape=gen_shape).flow()
        val_gen2 = benchmark.DataGenerator(loader, partition='val', batch_size=gParameters['batch_size'], shape=gen_shape).flow()
        test_gen = benchmark.DataGenerator(loader, partition='test', batch_size=gParameters['batch_size'], shape=gen_shape).flow()

        train_steps = int(loader.n_train/gParameters['batch_size'])
        val_steps = int(loader.n_val/gParameters['batch_size'])
        test_steps = int(loader.n_test/gParameters['batch_size'])

        if 'train_steps' in gParameters:
            train_steps = gParameters['train_steps']
        if 'val_steps' in gParameters:
            val_steps = gParameters['val_steps']
        if 'test_steps' in gParameters:
            test_steps = gParameters['test_steps']

        checkpointer = ModelCheckpoint(filepath=gParameters['output_dir']+'.model'+self.ext+'.h5', save_best_only=True)
        progbar = MyProgbarLogger(train_steps * gParameters['batch_size'])
        loss_history = MyLossHistory(progbar=progbar, val_gen=val_gen2, test_gen=test_gen,
                                val_steps=val_steps, test_steps=test_steps,
                                metric=gParameters['loss'], category_cutoffs=gParameters['category_cutoffs'],
                                ext=self.ext, pre=gParameters['output_dir'])

        # Seed random generator for training
        np.random.seed(seed)

        candleRemoteMonitor = candle.CandleRemoteMonitor(params=gParameters)

        print(loader.target_variable)

        start_time = time.time()

        history = model.fit_generator(train_gen, train_steps,
                            epochs=gParameters['epochs'],
                            validation_data=val_gen,
                            validation_steps=val_steps,
                            verbose=0,
                            callbacks=[checkpointer, loss_history, progbar, candleRemoteMonitor],
                            )

        end_time = time.time()
        self.runtime = round(end_time - start_time, 2)

        print("Training Time: ", self.runtime, " seconds")

        test_x = loader.test.drop([self.target_variable], 1).to_numpy()
        test_y = loader.test[self.target_variable].to_numpy()

        y_pred = model.predict(test_x)
        self.r2_error = round(r2_score(test_y, y_pred), 4)

        mse = mean_squared_error(test_y, y_pred)
        self.rmse = round(np.sqrt(mse), 4)



    def setup(self, gParameters):
        """
        Runs the model using the specified set of parameters

        Args:
        gParameters: a python dictionary containing the parameters (e.g. epoch)
        to run the model with.
        """
        #
        if 'dense' in gParameters:
            dval = gParameters['dense']
            if type(dval) != list:
                res = list(dval)
                gParameters['dense'] = res
            #print(gParameters['dense'])

        if 'conv' in gParameters:
            flat = gParameters['conv']
            gParameters['conv'] = [flat[i:i+3] for i in range(0, len(flat), 3)]
            #print('Conv input', gParameters['conv'])

        # Construct extension to save model
        self.ext = benchmark.extension_from_parameters(gParameters, '.keras')
        logfile = gParameters['logfile'] if gParameters['logfile'] else gParameters['output_dir']+self.ext+'.log'

        fh = logging.FileHandler(logfile)
        fh.setFormatter(logging.Formatter("[%(asctime)s %(process)d] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        fh.setLevel(logging.DEBUG)

        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter(''))
        sh.setLevel(logging.DEBUG if gParameters['verbose'] else logging.INFO)

        benchmark.logger.setLevel(logging.DEBUG)
        benchmark.logger.addHandler(fh)
        benchmark.logger.addHandler(sh)
        benchmark.logger.info('Params: {}'.format(gParameters))

        return fh, sh

    def define_model(self, gParameters, loader, initializer_weights, initializer_bias):
        # Define model architecture
        gen_shape = None
        out_dim = 1

        model = Sequential()
        if 'dense' in gParameters: # Build dense layers
            for layer in gParameters['dense']:
                if layer:
                    model.add(Dense(layer, input_dim=loader.input_dim,
                                kernel_initializer=initializer_weights,
                                bias_initializer=initializer_bias))
                    if gParameters['batch_normalization']:
                        model.add(BatchNormalization())
                    model.add(Activation(gParameters['activation']))
                    if gParameters['dropout']:
                        model.add(Dropout(gParameters['dropout']))
        else: # Build convolutional layers
            gen_shape = 'add_1d'
            layer_list = list(range(0, len(gParameters['conv'])))
            lc_flag=False
            if 'locally_connected' in gParameters:
                lc_flag = True

            for l, i in enumerate(layer_list):
                if i == 0:
                    add_conv_layer(model, gParameters['conv'][i], input_dim=loader.input_dim,locally_connected=lc_flag)
                else:
                    add_conv_layer(model, gParameters['conv'][i],locally_connected=lc_flag)
                if gParameters['batch_normalization']:
                        model.add(BatchNormalization())
                model.add(Activation(gParameters['activation']))
                if gParameters['pool']:
                    model.add(MaxPooling1D(pool_size=gParameters['pool']))
            model.add(Flatten())

        model.add(Dense(out_dim))

        return model, gen_shape

def rmse(y_true, y_pred):
    return K.sqrt(mean_squared_error(y_true, y_pred))

def initialize_parameters():

    # Build benchmark object
    p1b3Bmk = benchmark.BenchmarkP1B3(benchmark.file_path, CNN_CONFIG, 'keras',
    prog='p1b3_baseline', desc='Multi-task (DNN) for data extraction from clinical reports - Pilot 3 Benchmark 1')
    
    # Initialize parameters
    gParameters = candle.finalize_parameters(p1b3Bmk)
    #benchmark.logger.info('Params: {}'.format(gParameters))

    benchmark.check_params(gParameters)

    return gParameters

def str2lst(string_val):
    result = [int(x) for x in string_val.split(' ')]
    return result


def evaluate_keras_metric(y_true, y_pred, metric):
    objective_function = metrics.get(metric)
    objective = objective_function(y_true, y_pred)
    return K.eval(objective)


def evaluate_model(model, generator, steps, metric, category_cutoffs=[0.]):
    y_true, y_pred = None, None
    count = 0
    while count < steps:
        x_batch, y_batch = next(generator)
        y_batch_pred = model.predict_on_batch(x_batch)
        y_batch_pred = y_batch_pred.ravel()
        y_true = np.concatenate((y_true, y_batch)) if y_true is not None else y_batch
        y_pred = np.concatenate((y_pred, y_batch_pred)) if y_pred is not None else y_batch_pred
        count += 1

    loss = evaluate_keras_metric(y_true.astype(np.float32), y_pred.astype(np.float32), metric)

    y_true_class = np.digitize(y_true, category_cutoffs)
    y_pred_class = np.digitize(y_pred, category_cutoffs)

    # theano does not like integer input
    acc = evaluate_keras_metric(y_true_class.astype(np.float32), y_pred_class.astype(np.float32), 'binary_accuracy')  # works for multiclass labels as well

    return loss, acc, y_true, y_pred, y_true_class, y_pred_class


def plot_error(y_true, y_pred, batch, file_ext, file_pre='output_dir', subsample=1000):
    if batch % 10:
        return

    total = len(y_true)
    if subsample and subsample < total:
        usecols = np.random.choice(total, size=subsample, replace=False)
        y_true = y_true[usecols]
        y_pred = y_pred[usecols]

    y_true = y_true * 100
    y_pred = y_pred * 100
    diffs = y_pred - y_true

    bins = np.linspace(-200, 200, 100)
    if batch == 0:
        y_shuf = np.random.permutation(y_true)
        plt.hist(y_shuf - y_true, bins, alpha=0.5, label='Random')

    #plt.hist(diffs, bins, alpha=0.35-batch/100., label='Epoch {}'.format(batch+1))
    plt.hist(diffs, bins, alpha=0.3, label='Epoch {}'.format(batch+1))
    plt.title("Histogram of errors in percentage growth")
    plt.legend(loc='upper right')
    plt.savefig(file_pre+'.histogram'+file_ext+'.b'+str(batch)+'.png')
    plt.close()

    # Plot measured vs. predicted values
    fig, ax = plt.subplots()
    plt.grid('on')
    ax.scatter(y_true, y_pred, color='red', s=10)
    ax.plot([y_true.min(), y_true.max()],
            [y_true.min(), y_true.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.savefig(file_pre+'.diff'+file_ext+'.b'+str(batch)+'.png')
    plt.close()


class MyLossHistory(Callback):
    def __init__(self, progbar, val_gen, test_gen, val_steps, test_steps, metric, category_cutoffs=[0.], ext='', pre='save'):
        super(MyLossHistory, self).__init__()
        self.progbar = progbar
        self.val_gen = val_gen
        self.test_gen = test_gen
        self.val_steps = val_steps
        self.test_steps = test_steps
        self.metric = metric
        self.category_cutoffs = category_cutoffs
        self.pre = pre
        self.ext = ext

    def on_train_begin(self, logs={}):
        self.best_val_loss = np.Inf
        self.best_val_acc = -np.Inf

    def on_epoch_end(self, batch, logs={}):
        val_loss, val_acc, y_true, y_pred, y_true_class, y_pred_class = evaluate_model(self.model, self.val_gen, self.val_steps, self.metric, self.category_cutoffs)
        test_loss, test_acc, _, _, _, _ = evaluate_model(self.model, self.test_gen, self.test_steps, self.metric, self.category_cutoffs)
        self.progbar.append_extra_log_values([('val_acc', val_acc), ('test_loss', test_loss), ('test_acc', test_acc)])
        if float(logs.get('val_loss', 0)) < self.best_val_loss:
            plot_error(y_true, y_pred, batch, self.ext, self.pre)
        self.best_val_loss = min(float(logs.get('val_loss', 0)), self.best_val_loss)
        self.best_val_acc = max(float(logs.get('val_acc', 0)), self.best_val_acc)


class MyProgbarLogger(ProgbarLogger):
    def __init__(self, samples):
        super(MyProgbarLogger, self).__init__(count_mode='samples')
        self.samples = samples

    def on_train_begin(self, logs=None):
        super(MyProgbarLogger, self).on_train_begin(logs)
        self.verbose = 1
        self.extra_log_values = []
        self.params['samples'] = self.samples

    def on_batch_begin(self, batch, logs=None):
        if self.seen < self.target:
            self.log_values = []
            self.extra_log_values = []

    def append_extra_log_values(self, tuples):
        for k, v in tuples:
            self.extra_log_values.append((k, v))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        epoch_log = 'Epoch {}/{}'.format(epoch + 1, self.epochs)
        for k in self.params['metrics']:
            if k in logs:
                self.log_values.append((k, logs[k]))
                epoch_log += ' - {}: {:.4f}'.format(k, logs[k])
        for k, v in self.extra_log_values:
            self.log_values.append((k, v))
            epoch_log += ' - {}: {:.4f}'.format(k, float(v))
        if self.verbose:
            self.progbar.update(self.seen, self.log_values)
        benchmark.logger.debug(epoch_log)

def add_conv_layer(model, layer_params, input_dim=None, locally_connected=False):
    if len(layer_params) == 3: # 1D convolution
        filters = layer_params[0]
        filter_len = layer_params[1]
        stride = layer_params[2]
        if locally_connected:
            if input_dim:
                model.add(LocallyConnected1D(filters, filter_len, strides=stride, input_shape=(input_dim, 1)))
            else:
                model.add(LocallyConnected1D(filters, filter_len, strides=stride))
        else:
            if input_dim:
                model.add(Conv1D(filters, filter_len, strides=stride, input_shape=(input_dim, 1)))
            else:
                model.add(Conv1D(filters, filter_len, strides=stride))
    elif len(layer_params) == 5: # 2D convolution
        filters = layer_params[0]
        filter_len = (layer_params[1], layer_params[2])
        stride = (layer_params[3], layer_params[4])
        if locally_connected:
            if input_dim:
                model.add(LocallyConnected2D(filters, filter_len, strides=stride, input_shape=(input_dim, 1)))
            else:
                model.add(LocallyConnected2D(filters, filter_len, strides=stride))
        else:
            if input_dim:
                model.add(Conv2D(filters, filter_len, strides=stride, input_shape=(input_dim, 1)))
            else:
                model.add(Conv2D(filters, filter_len, strides=stride))
    return model

