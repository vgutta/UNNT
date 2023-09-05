import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import argparse

DATA_PATH = os.path.join(os.path.dirname(__file__), '../../', 'data')

def load_and_preprocess_default_data():

    lincs1000_genes = pd.read_csv(os.path.join(DATA_PATH, 'lincs1000.tsv'), sep='\t')
    nci_exp = pd.read_csv(os.path.join(DATA_PATH, 'cell_exp_nci.tsv'), sep= '\t')
    nci_fda_drugs = pd.read_csv(os.path.join(DATA_PATH, 'nci_fda_drugs.csv'), sep='\t')

    nci_fda_drugs['NSC'] = nci_fda_drugs['NSC'].astype(str)
    
    lincs_index = lincs1000_genes.set_index(['symbol']).index

    # filter pandas data frame columns by list and include index column
    nci_exp = nci_exp.loc[:, nci_exp.columns.isin(lincs_index) | nci_exp.columns.isin(['CELLNAME'])]

    # format CELLNAME column
    nci_exp['CELLNAME'] = nci_exp['CELLNAME'].str.split(':', n=1).str[1]

    nci_exp['CELLNAME'] = nci_exp['CELLNAME'].str.replace('_', '')

    nci_fda_dose_response = pd.read_csv(os.path.join(DATA_PATH, 'nci_fda_drug_response.tsv'), sep='\t', converters ={'DRUG' : str})

    nci_merged_data = nci_fda_dose_response.merge(nci_exp, on='CELLNAME')

    #sample_drugs = set(nci_merged_data['DRUG'])

    drug_desc = pd.read_csv(os.path.join(DATA_PATH, 'fda_drug_desc.tsv'), sep='\t', engine='c',
                        na_values=['na','-',''],
                        converters ={'NAME' : str})
    drug_desc.rename(columns={'NAME': 'DRUG'}, inplace=True)
    drug_desc = drug_desc.dropna()

    #drug_desc = drug_desc.drop(drug_desc.columns[1000:], axis=1)

    # reduce nci_merged_data rows to only 10% of the original to complete faster
    nci_merged_data = nci_merged_data.sample(frac=0.1)

    nci_merged_data = nci_merged_data.merge(drug_desc, on='DRUG')

    all_data = nci_merged_data

    auc_values = all_data['AUC']

    all_data = all_data.drop(columns=['CELLNAME', 'DRUG', 'AUC'])

    x_train, x_test, y_train, y_test = train_test_split(all_data, auc_values, test_size=0.3)

    x_train = x_train.to_numpy()
    x_test = x_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    return x_train, x_test, y_train, y_test


def load_and_preprocess_custom_data(args):

    user_data = pd.read_csv(os.path.join(DATA_PATH, args.data_file), sep=',')

    y_values = user_data[args.target_variable]

    x_train, x_test, y_train, y_test = train_test_split(user_data, y_values, test_size=args.test_size)

    x_train = x_train.to_numpy()
    x_test = x_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    return x_train, x_test, y_train, y_test