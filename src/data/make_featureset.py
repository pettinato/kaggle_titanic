# -*- coding: utf-8 -*-
# Run script as python3 -m make_featureset
# since it's inside of this module.

import os

# Having trouble with this '-m' didn't help
from ..features import build_features as bf


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
        os.path.abspath(__file__)), '../../'))
    input = os.path.join(proj_dir, 'data/raw/train.csv')
    output = os.path.join(proj_dir, 'data/raw/feature_set.csv')
    print('Reading input data from {}'.format(input))
    print('Writing Features to file {}'.format(output))
    main(input, output)
