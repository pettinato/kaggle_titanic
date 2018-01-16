# -*- coding: utf-8 -*-

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.features import build_features as bf


def main(input_filepath, output_filepath):
    """
    Build features and store for further processing.
    :param input_filepath: Raw input Titanic training set.
    :param output_filepath: Output filename for constructed features
    """
    raw_df = bf.read_raw_data(input_filepath)
    features_df = bf.build_features(raw_df)
    features_df.to_csv(output_filepath, index=False)


if __name__ == '__main__':
    proj_dir = os.path.abspath(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), '../'))
    input_filename = os.path.join(proj_dir, 'data/raw/train.csv')
    output_filename = os.path.join(proj_dir, 'data/processed/feature_set.csv')
    print('Reading input data from {}'.format(input_filename))
    print('Writing Features to file {}'.format(output_filename))
    main(input_filename, output_filename)
    print('Finished writing features')
