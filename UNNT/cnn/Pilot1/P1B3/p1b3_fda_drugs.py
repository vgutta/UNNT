from __future__ import absolute_import

import collections
import gzip
import logging
import os
import sys
import multiprocessing
import threading
import argparse
try:
    import configparser
except ImportError:
    import ConfigParser as configparser

import numpy as np
import pandas as pd

from itertools import cycle, islice

try:
    from sklearn.impute import SimpleImputer as Imputer
except ImportError:
    from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler

file_path = os.path.dirname(os.path.realpath(__file__))
lib_path = os.path.abspath(os.path.join(file_path, '..' ))
sys.path.append(lib_path)
lib_path2 = os.path.abspath(os.path.join(file_path, '..', '..', 'common'))
sys.path.append(lib_path2)
print(sys.path)

DATA_PATH = os.path.join(os.path.dirname(__file__), '../../../../', 'data')

import candle

logger = logging.getLogger(__name__)

# Number of data generator workers
WORKERS = 1

class BenchmarkP1B3(candle.Benchmark):

    def set_locals(self):
        """Functionality to set variables specific for the benchmark
        - required: set of required parameters for the benchmark.
        - additional_definitions: list of dictionaries describing the additional parameters for the
        benchmark.
        """

        if required is not None:
            self.required = set(required)
        if additional_definitions is not None:
            self.additional_definitions = additional_definitions

additional_definitions = [
# Feature selection
    {'name':'cell_features', 
        'nargs':'+',
        #'default':'argparse.SUPPRESS',
        'choices':['expression', 'mirna', 'proteome', 'all', 'categorical'],
        'help':'use one or more cell line feature sets: "expression", "mirna", "proteome", "all"; or use "categorical" for one-hot encoding of cell lines'},
    {'name':'drug_features', 
        'nargs':'+',
        #'default':'argparse.SUPPRESS',
        'choices':['descriptors', 'latent', 'all', 'noise'],
        'help':"use dragon7 descriptors, latent representations from Aspuru-Guzik's SMILES autoencoder, or both, or random features; 'descriptors','latent', 'all', 'noise'"},
    {'name':'cell_noise_sigma', 'type':float,
        'help':"standard deviation of guassian noise to add to cell line features during training"},
# Output selection
    {'name':'min_logconc', 
        'type':float,
        #'default':'argparse.SUPPRESS',
        'help':"min log concentration of dose response data to use: -3.0 to -7.0"},
    {'name':'max_logconc',  
        'type':float,
        #'default':'argparse.SUPPRESS',
        'help':"max log concentration of dose response data to use: -3.0 to -7.0"},
    {'name':'subsample',
        #'default':'argparse.SUPPRESS',
        'choices':['naive_balancing', 'none'],
        'help':"dose response subsample strategy; 'none' or 'naive_balancing'"},
    {'name':'category_cutoffs', 
        'nargs':'+', 
        'type':float,
        #'default':'argparse.SUPPRESS',
        'help':"list of growth cutoffs (between -1 and +1) seperating non-response and response categories"},
# Sample data selection
    {'name':'test_cell_split', 
        'type':float,
        #'default':'argparse.SUPPRESS',
        'help':"cell lines to use in test; if None use predefined unseen cell lines instead of sampling cell lines used in training"},
# Test random model
    {'name':'scramble', 
        'type': candle.str2bool, 
        'default': False, 
        'help':'randomly shuffle dose response data'},
    {'name':'workers', 
        'type':int,
        'default':WORKERS,
        'help':'number of data generator workers'}
]

required = [
    'activation',
    'batch_size',
    'batch_normalization',
    'category_cutoffs',
    'cell_features',
    'dropout',
    'drug_features',
    'epochs',
    'feature_subsample',
    'initialization',
    'learning_rate',
    'loss',
    'min_logconc',
    'max_logconc',
    'optimizer',
#    'penalty',
    'rng_seed',
    'scaling',
    'subsample',
    'test_cell_split',
    'val_split',
    'cell_noise_sigma'
    ]

def check_params(fileParams):
    # Allow for either dense or convolutional layer specification
    # if none found exit
    try:
        fileParams['dense']
    except KeyError:
        try: 
            fileParams['conv'] 
        except KeyError: 
            print("Error! No dense or conv layers specified. Wrong file !! ... exiting ")
            raise
        else:
            try:
                fileParams['pool']
            except KeyError:
                fileParams['pool'] = None
                print("Warning ! No pooling specified after conv layer.")


def extension_from_parameters(params, framework):
    """Construct string for saving model with annotation of parameters"""
    ext = framework
    ext += '.A={}'.format(params['activation'])
    ext += '.B={}'.format(params['batch_size'])
    ext += '.D={}'.format(params['dropout'])
    ext += '.E={}'.format(params['epochs'])
    if params['feature_subsample']:
        ext += '.F={}'.format(params['feature_subsample'])
    if 'cell_noise_sigma' in params:
        ext += '.N={}'.format(params['cell_noise_sigma'])
    if 'conv' in params:
        name = 'LC' if 'locally_connected' in params else 'C'
        layer_list = list(range(0, len(params['conv'])))
        for l, i in enumerate(layer_list):
            filters = params['conv'][i][0]
            filter_len = params['conv'][i][1]
            stride = params['conv'][i][2]
            if filters <= 0 or filter_len <= 0 or stride <= 0:
                break
            ext += '.{}{}={},{},{}'.format(name, l+1, filters, filter_len, stride)
        if 'pool' in params and params['conv'][0] and params['conv'][1]:
            ext += '.P={}'.format(params['pool'])
    if 'dense' in params:
        for i, n in enumerate(params['dense']):
            if n:
                ext += '.D{}={}'.format(i+1, n)
    if params['batch_normalization']:
        ext += '.BN'
    ext += '.S={}'.format(params['scaling'])

    return ext


def scale(df, scaling=None):
    """Scale data included in pandas dataframe.

    Parameters
    ----------
    df : pandas dataframe
        dataframe to scale
    scaling : 'maxabs', 'minmax', 'std', or None, optional (default 'std')
        type of scaling to apply
    """

    if scaling is None or scaling.lower() == 'none':
        return df

    df = df.dropna(axis=1, how='any')

    # Scaling data
    if scaling == 'maxabs':
        # Normalizing -1 to 1
        scaler = MaxAbsScaler()
    elif scaling == 'minmax':
        # Scaling to [0,1]
        scaler = MinMaxScaler()
    else:
        # Standard normalization
        scaler = StandardScaler()

    mat = df.as_matrix()
    mat = scaler.fit_transform(mat)

    df = pd.DataFrame(mat, columns=df.columns)

    return df


def impute_and_scale(df, scaling='std'):
    """Impute missing values with mean and scale data included in pandas dataframe.

    Parameters
    ----------
    df : pandas dataframe
        dataframe to impute and scale
    scaling : 'maxabs' [-1,1], 'minmax' [0,1], 'std', or None, optional (default 'std')
        type of scaling to apply
    """

    df = df.dropna(axis=1, how='all')

    #imputer = Imputer(strategy='mean', axis=0)
    imputer = Imputer(strategy='mean')
    mat = imputer.fit_transform(df)

    if scaling is None or scaling.lower() == 'none':
        return pd.DataFrame(mat, columns=df.columns)

    if scaling == 'maxabs':
        scaler = MaxAbsScaler()
    elif scaling == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    mat = scaler.fit_transform(mat)

    df = pd.DataFrame(mat, columns=df.columns)

    return df


def load_cellline_expressions():
    """Load cell line expression data, sub-select columns of gene expression
        randomly if specificed, scale the selected data and return a
        pandas dataframe.

    Parameters
    ----------
    path: string
        path to 'cell_exp_nci.tsv'
    dtype: numpy type
        precision (data type) for reading float values
    ncols : int or None
        number of columns (gene expression) to randomly subselect (default None : use all data)
    scaling : 'maxabs' [-1,1], 'minmax' [0,1], 'std', or None, optional (default 'std')
        type of scaling to apply
    """
    
    lincs1000_genes = pd.read_csv(os.path.join(DATA_PATH,'lincs1000.tsv'), sep='\t')
    nci_exp = pd.read_csv(os.path.join(DATA_PATH,'cell_exp_nci.tsv'), sep= '\t')
    
    lincs_index = lincs1000_genes.set_index(['symbol']).index

    # filter pandas data frame columns by list and include index column
    nci_exp = nci_exp.loc[:, nci_exp.columns.isin(lincs_index) | nci_exp.columns.isin(['CELLNAME'])]

    # format CELLNAME column
    nci_exp['CELLNAME'] = nci_exp['CELLNAME'].str.split(':', n=1).str[1]

    nci_exp['CELLNAME'] = nci_exp['CELLNAME'].str.replace('_', '')
    
    nci_exp.rename(columns={'DRUG':'NSC'}, inplace=True)
    
    nci_exp = nci_exp.dropna()

    nci_exp = nci_exp.sample(frac=0.1)

    return nci_exp

    #return df

def load_drug_descriptors():
    """Load drug descriptor data, sub-select columns of drugs descriptors
        randomly if specificed, impute and scale the selected data, and return a
        pandas dataframe.

    Parameters
    ----------
    path: string
        path to 'fda_drug_desc.tsv'
    dtype: numpy type
        precision (data type) for reading float values
    ncols : int or None
        number of columns (drugs descriptors) to randomly subselect (default None : use all data)
    scaling : 'maxabs' [-1,1], 'minmax' [0,1], 'std', or None, optional (default 'std')
        type of scaling to apply
    """
    
    drug_desc = pd.read_csv(os.path.join(DATA_PATH, 'fda_drug_desc.tsv'), sep='\t', engine='c',
                     na_values=['na','-',''],
                     converters ={'NAME' : str})
    drug_desc.rename(columns={'NAME': 'NSC'}, inplace=True)

    #drug_desc = drug_desc.drop(drug_desc.columns[1000:], axis=1)
    
    drug_desc = drug_desc.dropna()
        
    return drug_desc
    

def load_dose_response():
    """Load cell line response to different drug compounds, sub-select response for a specific
        drug log concentration range and return a pandas dataframe.

    Parameters
    ----------
    path: string
        path to 'nci_fda_drug_response.tsv'
    seed: integer
        seed for random generation
    dtype: numpy type
        precision (data type) for reading float values
    min_logconc : -3, -4, -5, -6, -7, optional (default -5)
        min log concentration of drug to return cell line growth
    max_logconc : -3, -4, -5, -6, -7, optional (default -5)
        max log concentration of drug to return cell line growth
    subsample: None, 'naive_balancing' (default None)
        subsampling strategy to use to balance the data based on growth
    """

    df = pd.read_csv(os.path.join(DATA_PATH, 'nci_fda_drug_response.tsv'), sep='\t', converters ={'DRUG' : str})
    df.rename(columns={'DRUG':'NSC'}, inplace=True)
    df = df.set_index(['NSC'])

    return df

class DataLoader(object):
    """Load merged drug response, drug descriptors and cell line essay data
    """

    def __init__(self, seed, dtype, val_split=0.2, test_cell_split=None, shuffle=True,
                 cell_features=['expression'], drug_features=['descriptors'],
                 feature_subsample=None, scaling='std', scramble=False,
                 min_logconc=-5., max_logconc=-4., subsample='naive_balancing',
                 category_cutoffs=[0.]):
        """Initialize data merging drug response, drug descriptors and cell line essay.
           Shuffle and split training and validation set

        Parameters
        ----------
        seed: integer
            seed for random generation
        dtype: numpy type
            precision (data type) for reading float values
        val_split : float, optional (default 0.2)
            fraction of data to use in validation
        test_cell_split : float or None, optional (default None)
            fraction of cell lines to use in test; if None use predefined unseen cell lines instead of sampling cell lines used in training
        shuffle : True or False, optional (default True)
            if True shuffles the merged data before splitting training and validation sets
        cell_features: list of strings from 'expression', 'mirna', 'proteome', 'all', 'categorical' (default ['expression'])
            use one or more cell line feature sets: gene expression, microRNA, proteomics; or, use 'categorical' for one-hot encoded cell lines
        drug_features: list of strings from 'descriptors', 'latent', 'all', 'noise' (default ['descriptors'])
            use dragon7 descriptors, latent representations from Aspuru-Guzik's SMILES autoencoder trained on NSC drugs, or both; use random features if set to noise
        feature_subsample: None or integer (default None)
            number of feature columns to use from cellline expressions and drug descriptors
        scaling: None, 'std', 'minmax' or 'maxabs' (default 'std')
            type of feature scaling: 'maxabs' to [-1,1], 'maxabs' to [-1, 1], 'std' for standard normalization
        scramble: True or False, optional (default False)
            if True randomly shuffle dose response data as a control
        min_logconc: float value between -3 and -7, optional (default -5.)
            min log concentration of drug to return cell line growth
        max_logconc: float value between -3 and -7, optional (default -4.)
            max log concentration of drug to return cell line growth
        subsample: 'naive_balancing' or None
            if True balance dose response data with crude subsampling
        category_cutoffs: list of floats (between -1 and +1) (default None)
            growth thresholds seperating non-response and response categories
        """

        #cell_expr_path, cell_mrna_path, cell_prot_path, cell_kino_path,drug_desc_path, drug_auen_path, dose_resp_path, test_cell_path, test_drug_path = stage_data()
        # Seed random generator for loading data
        np.random.seed(seed)

        df = load_dose_response()
        logger.info('Loaded {} unique (D, CL) response sets.'.format(df.shape[0]))
        # df[['GROWTH', 'LOG_CONCENTRATION']].to_csv('all.response.csv')
        df = df.reset_index()

        if 'all' in cell_features:
            self.cell_features = ['expression', 'mirna', 'proteome']
        else:
            self.cell_features = cell_features

        if 'all' in drug_features:
            self.drug_features = ['descriptors', 'latent']
        else:
            self.drug_features = drug_features

        self.input_shapes = collections.OrderedDict()
        #self.input_shapes['drug_concentration'] = (1,)

        for fea in self.cell_features:
            if fea == 'expression':
                self.df_cell_expr = load_cellline_expressions()
                self.input_shapes['cell_expression'] = (self.df_cell_expr.shape[1] - 1,)
                df = df.merge(self.df_cell_expr[['CELLNAME']], on='CELLNAME')       

        for fea in self.drug_features:
            if fea == 'descriptors':
                self.df_drug_desc = load_drug_descriptors()
                self.input_shapes['drug_descriptors'] = (self.df_drug_desc.shape[1] - 1,)
                df = df.merge(self.df_drug_desc[['NSC']], on='NSC')

        logger.debug('Filtered down to {} rows with matching information.'.format(df.shape[0]))

        df_test_cell = pd.read_csv(os.path.join(DATA_PATH, 'val_nci60_cell.csv'), sep=',', dtype={'CELLNAME':str})
        df_test_drug = pd.read_csv(os.path.join(DATA_PATH, 'val_nci60_fda_drugs.csv'), sep=',', dtype={'NSC': object})

        df_train_val = df[(~df['NSC'].isin(df_test_drug['NSC'])) & (~df['CELLNAME'].isin(df_test_cell['CELLNAME']))]

        logger.debug('Combined train and validation set has {} rows'.format(df_train_val.shape[0]))

        if test_cell_split and test_cell_split > 0:
            df_test_cell = df_train_val[['CELLNAME']].drop_duplicates().sample(frac=test_cell_split, random_state=seed)
            logger.debug('Use unseen drugs and a fraction of seen cell lines for testing: ' + ', '.join(sorted(list(df_test_cell['CELLNAME']))))
        else:
            logger.debug('Use unseen drugs and predefined unseen cell lines for testing: ' + ', '.join(sorted(list(df_test_cell['CELLNAME']))))
            
        self.df_test = df.merge(df_test_cell, on='CELLNAME').merge(df_test_drug, on='NSC')
        
        logger.debug('Test set has {} rows'.format(self.df_test.shape[0]))

        if shuffle:
            df_train_val = df_train_val.sample(frac=1.0, random_state=seed)
            self.df_test = self.df_test.sample(frac=1.0, random_state=seed)

        self.df_response = pd.concat([df_train_val, self.df_test]).reset_index(drop=True)

        if scramble:
            growth = self.df_response[['GROWTH']]
            random_growth = growth.iloc[np.random.permutation(np.arange(growth.shape[0]))].reset_index()
            self.df_response[['GROWTH']] = random_growth['GROWTH']
            logger.warn('Randomly shuffled dose response growth values.')

        logger.info('Distribution of dose response:')
        logger.info(self.df_response[['AUC']].describe())

        if category_cutoffs is not None:
            growth = self.df_response['AUC']
            classes = np.digitize(growth, category_cutoffs)
            bc = np.bincount(classes)
            min_g = np.min(growth) / 100
            max_g = np.max(growth) / 100
            logger.info('Category cutoffs: {}'.format(category_cutoffs))
            logger.info('Dose response bin counts:')
            for i, count in enumerate(bc):
                lower = min_g if i == 0 else category_cutoffs[i-1]
                upper = max_g if i == len(bc)-1 else category_cutoffs[i]
                logger.info('  Class {}: {:7d} ({:.4f}) - between {:+.2f} and {:+.2f}'.
                            format(i, count, count/len(growth), lower, upper))
            logger.info('  Total: {:9d}'.format(len(growth)))

        self.total = df_train_val.shape[0]
        self.n_test = self.df_test.shape[0]
        self.n_val = int(self.total * val_split)
        self.n_train = self.total - self.n_val
        logger.info('Rows in train: {}, val: {}, test: {}'.format(self.n_train, self.n_val, self.n_test))

        logger.info('Input features shapes:')
        for k, v in self.input_shapes.items():
            logger.info('  {}: {}'.format(k, v))

        self.input_dim = sum([np.prod(x) for x in self.input_shapes.values()])
        logger.info('Total input dimensions: {}'.format(self.input_dim))


class DataGenerator(object):
    """Generate training, validation or testing batches from loaded data
    """

    def __init__(self, data, partition='train', batch_size=32, shape=None, concat=True, cell_noise_sigma=None):
        """Initialize data

        Parameters
        ----------
        data: DataLoader object
            loaded data object containing original data frames for molecular, drug and response data
        partition: 'train', 'val', or 'test'
            partition of data to generate for
        batch_size: integer (default 32)
            batch size of generated data
        shape: None, '1d' or 'add_1d' (default None)
            keep original feature shapes, make them flat or add one extra dimension (for convolution or locally connected layers in some frameworks)
        concat: True or False (default True)
            concatenate all features if set to True
        cell_noise_sigma: float
            standard deviation of guassian noise to add to cell line features during training
        """
        self.lock = threading.Lock()
        self.data = data
        self.partition = partition
        self.batch_size = batch_size
        self.shape = shape
        self.concat = concat
        self.cell_noise_sigma = cell_noise_sigma

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
        """Keep generating data batches
        """
        while 1:
            self.lock.acquire()
            indices = list(islice(self.cycle, self.batch_size))
            self.lock.release()

            df = self.data.df_response.iloc[indices, :]
            cell_column_beg = df.shape[1]

            for fea in self.data.cell_features:
                if fea == 'expression':
                    df = pd.merge(df, self.data.df_cell_expr, on='CELLNAME')

            cell_column_end = df.shape[1]

            for fea in self.data.drug_features:
                if fea == 'descriptors':
                    df = df.merge(self.data.df_drug_desc, on='NSC')

            df = df.drop(['CELLNAME', 'NSC'], 1)
            x = np.array(df.iloc[:, 1:])

            if self.cell_noise_sigma:
                c1 = cell_column_beg - 3
                c2 = cell_column_end - 3
                x[:, c1:c2] += np.random.randn(df.shape[0], c2-c1) * self.cell_noise_sigma

            y = np.array(df.iloc[:, 0])
            y = y / 100.

            if self.concat:
                if self.shape == 'add_1d':
                    yield x.reshape(x.shape + (1,)), y
                else:
                    yield x, y
            else:
                x_list = []
                index = 0
                for v in self.data.input_shapes.values():
                    length = np.prod(v)
                    subset = x[:, index:index+length]
                    if self.shape == '1d':
                        reshape = (x.shape[0], length)
                    elif self.shape == 'add_1d':
                        reshape = (x.shape[0],) + v + (1,)
                    else:
                        reshape = (x.shape[0],) + v
                    x_list.append(subset.reshape(reshape))
                    index += length
                yield x_list, y