import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import argparse


def load_and_preprocess_default_data():

    lincs1000_genes = pd.read_csv('./data/lincs1000.tsv', sep='\t')
    nci_exp = pd.read_csv('./data/cell_exp_nci.tsv', sep= '\t')
    nci_fda_drugs = pd.read_csv('./data/nci_fda_drugs.csv', sep='\t')

    nci_fda_drugs['NSC'] = nci_fda_drugs['NSC'].astype(str)

    fda_set = None

    lincs_index = lincs1000_genes.set_index(['symbol']).index

    # filter pandas data frame columns by list and include index column
    nci_exp = nci_exp.loc[:, nci_exp.columns.isin(lincs_index) | nci_exp.columns.isin(['CELLNAME'])]

    # format CELLNAME column
    nci_exp['CELLNAME'] = nci_exp['CELLNAME'].str.split(':', n=1).str[1]

    nci_exp['CELLNAME'] = nci_exp['CELLNAME'].str.replace('_', '')

    combined_dose_response = pd.read_csv('./data/combined_single_response_agg', sep='\t')

    nci60_dose_response = combined_dose_response[combined_dose_response['SOURCE'] == 'NCI60']

    nci60_filtered = nci60_dose_response[['SOURCE', 'CELL', 'DRUG', 'AUC', 'IC50']]
    nci60_filtered.rename(columns={'CELL':'CELLNAME'}, inplace=True)
    nci60_filtered['DRUG'] = nci60_filtered['DRUG'].str.replace('NSC.', '')

    nci60_filtered['CELLNAME'] = nci60_filtered['CELLNAME'].str.replace('NCI60.', '')

    nci60_filtered['CELLNAME'] = nci60_filtered['CELLNAME'].str.replace('-', '')

    fda_set = nci60_filtered[nci60_filtered['DRUG'].isin(set(nci_fda_drugs['NSC']))]

    nci_merged_data = fda_set.merge(nci_exp, on='CELLNAME')

    sample_drugs = set(nci_merged_data['DRUG'])

    drug_desc = pd.read_csv('./data/descriptors.2D-NSC.5dose.filtered.txt', sep='\t', engine='c',
                        na_values=['na','-',''],
                        converters ={'NAME' : str})
    drug_desc.rename(columns={'NAME': 'NSC'}, inplace=True)

    subset_drug_desc = drug_desc[drug_desc['NSC'].isin(sample_drugs)]

    ## filter drug desc using filtered drug list
    subset_drug_desc.rename(columns={'NSC':'DRUG'}, inplace=True)

    nci_merged_data = nci_merged_data.merge(subset_drug_desc, on='DRUG')

    all_data = nci_merged_data

    auc_values = all_data['AUC']

    all_data_x = all_data.drop(columns=['CELLNAME', 'SOURCE', 'DRUG', 'AUC', 'IC50'])

    x_train, x_test, y_train, y_test = train_test_split(all_data_x, auc_values, test_size=0.3)

    x_train = x_train.to_numpy()
    x_test = x_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    return x_train, x_test, y_train, y_test


def load_and_preprocess_custom_data(args):

    user_data = pd.read_csv(args.path, sep=',')

    y_values = user_data[args.target_variable]

    x_train, x_test, y_train, y_test = train_test_split(user_data, y_values, test_size=args.test_size)

    x_train = x_train.to_numpy()
    x_test = x_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    return x_train, x_test, y_train, y_test