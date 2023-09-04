import threading
import argparse
import logging
import collections
from itertools import cycle, islice
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataLoader(object):
    """ Load and create a DataLoader object with user provided data"""

    def __init__(self, args):

        self.target_variable = args.target_variable
        self.user_data = pd.read_csv(args.path, sep=',')
        self.y_values = self.user_data[self.target_variable]

        self.input_shapes = collections.OrderedDict()

        df_test = self.user_data.sample(frac=args.test_split) # sample percent
        df_train_val = self.user_data.drop(df_test.index) # drop rows in training set

        self.total = df_train_val.shape[0]
        self.n_test = df_test.shape[0]
        self.n_val = int(self.total * args.val_split)
        self.n_train = self.total - self.n_val

        logger.info('Rows in train: {}, val: {}, test: {}'.format(self.n_train, self.n_val, self.n_test))

        self.input_dim = self.n_train.shape[1] - 1
        logger.info('Total input dimensions: {}'.format(self.input_dim))



class DataGenerator(object):
    """Generate training, validation or testing batches from loaded data"""

    def __init__(self, data, partition='train', batch_size=32, shape=None):
        """Initialize data

        Parameters
        ----------
        data: DataLoader object
            loaded data object containing original data frames for molecular, drug and response data
        partition: 'train', 'val', or 'test'
            partition of data to generate for
        batch_size: integer (default 32)
            batch size of generated data
        """

        self.lock = threading.Lock()
        self.data = data
        self.partition = partition
        self.batch_size = batch_size
        self.shape = shape

        if partition == 'train':
            self.cycle = cycle(range(data.n_train))
            self.num_data = data.n_train
        elif partition == 'val':
            self.cycle = cycle(range(data.total)[-data.n_val:])
            self.num_data = data.n_val
        elif partition == 'test':
            self.cycle = cycle(range(data.total, data.total + data.n_test))
            self.num_data = data.n_test
        else:
            raise Exception('Data partition "{}" not recognized.'.format(partition))


    def flow(self):
        """Keep generating data batches"""
        
        while 1:
            #self.lock.acquire()
            indices = list(islice(self.cycle, self.batch_size))
            #self.lock.release()

            df = self.data.iloc[indices, :]

            y = np.array(df[[self.data.target_variable]])
            y = y / 100

            x = np.array(df.drop([self.data.target_variable]))

            if self.shape == 'add_1d':
                yield x.reshape(x.shape + (1,)), y
            else:
                yield x, y