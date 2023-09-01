#imports
import argparse
import configparser
from create_tree import Tree
from p1b3.Pilot1.P1B3.cnn import CNN


def main(args):

    tree_model = Tree(args)
    tree_model.train()
    tree_model.evaluate()

    cnn = CNN()


if __name__ == '__main__':

    # initialize config parser to read from tree_config.txt
    config = configparser.ConfigParser()
    config.read('tree_config.txt')

    #code for flags
    parser = argparse.ArgumentParser(description='UNNT - Universal Neural Network Trainer')
    # default data args
    parser.add_argument('--data', type=str, default='nci60', help='path to data directory')
    parser.add_argument('--models', type=str, default='xgb', help='models to train')
    parser.add_argument('--gpu', action='store_true', help='Enable training on GPU')
    parser.add_argument('--target_variable', type=str, default=config.get('config', 'target_variable'), help='Predictor/Target/Y variable')
    args = parser.parse_args()
    print(args)

    main(args)

    
