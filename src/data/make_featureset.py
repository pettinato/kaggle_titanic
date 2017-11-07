# -*- coding: utf-8 -*-
import os
from src.features import build_features


def main(input_filepath, output_filepath):
    """
    Build features and store for further processing.
    :param input_filepath: Raw input Titanic training set.
    :param output_filepath: Output filename for constructed features
    """
    raw_df = build_features.read_raw_data(input_filepath)
    features_df = build_features.build_features(raw_df)
    features_df.to_csv(output_filepath, index=False)


if __name__ == '__main__':
    print(__file__)

    os.basename()

    input_filepath = '../../data/raw/train.csv'
    output_filepath = '../../data/processed/feature_set.csv'
    #main(input_filepath, output_filepath)
