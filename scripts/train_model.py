# -*- coding: utf-8 -*-

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models import train_model as tm


def main(input_filepath, output_filepath):
    """
    Build final model from raw training data.
    :param input_filepath: Raw input Titanic training set.
    :param output_filepath: Output filename for constructed model
    """
    tm.train_selected_random_forest(input_filename, output_filepath)


if __name__ == '__main__':
    proj_dir = os.path.abspath(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), '../'))
    input_filename = os.path.join(proj_dir, 'data/raw/train.csv')
    model_output_filename = os.path.join(proj_dir, 'models/random_forest_v1.pkl')
    print('Reading input data from {}'.format(input_filename))
    print('Writing Model to file {}'.format(model_output_filename))
    main(input_filename, model_output_filename)
    print('Finished training final model')
