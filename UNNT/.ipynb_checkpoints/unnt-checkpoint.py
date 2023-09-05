import ast
import argparse
import configparser
from xgb.create_tree import Tree
from cnn.Pilot1.P1B3.cnn import CNN


def main(args):

    tree_model = Tree(args)
    tree_model.train()
    tree_model.evaluate()

    cnn = CNN(args)

    print('\n############ FINAL RESULTS #############\n')

    results = [
        ['XGBoost', tree_model.rmse, tree_model.r2_error, tree_model.runtime],
        ['CNN', cnn.rmse, cnn.r2_error, cnn.runtime]
    ]

    # Calculate the maximum width for each column
    col_widths = [max(len(str(item)) + 2 for item in column) for column in zip(*results)]

    # Print the table headers
    print("Model".ljust(col_widths[0]) + "RMSE".ljust(col_widths[1]) + "R-squared".ljust(col_widths[2]) + "Training Time".ljust(col_widths[3]))

    # Print a separator line
    print("-" * (col_widths[0] + col_widths[1] + col_widths[2] + col_widths[3]))

    # Print the data
    for row in results:
        model, rmse, r2_error, training = row
        print(str(model).ljust(col_widths[0]) + str(rmse).ljust(col_widths[1]) + str(r2_error).ljust(col_widths[2]) + str(training).ljust(col_widths[3]))



if __name__ == '__main__':

    # initialize config parser to read from tree_config.txt
    config = configparser.ConfigParser()
    config.read('tree_config.txt')

    parser = argparse.ArgumentParser(description='UNNT - Universal Neural Network Trainer')
    parser.add_argument('--gpu', action='store_true', help='Enable training on GPU')

    for key, value in config.items('config'):
        parser.add_argument(f'--{key}', default=ast.literal_eval(value), help=f'{key} (default: {value})')

    args = parser.parse_args()

    main(args)

    
