# -*- coding: utf-8 -*-

"""As per https://www.kaggle.com/c/titanic#evaluation
the output file should have two columns,

PassengerId,Survived
892,0
893,1
894,0
"""

import os
import sys
from sklearn.externals import joblib
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.features import build_features as bf


def main(test_path, model_path, output_path):
    """
    Evaluate the final model on the test set
    :param test_path: Raw input Titanic test set.
    :param model_path: Input of the model joblib-pickled filename
    :param output_path: Path to store the predicted output data
    """
    raw_test_df = bf.read_raw_data(test_path)

    # TODO won't work, build_features assumes column Survived
    # SUGGEST UPDATE BUILD_FEATURES TO NOT ASSUME THIS COLUMN
    # requires updating the unit tests
    test_df = bf.build_features(test_path)

    # Verify output model
    loaded_model = joblib.load(model_path)
    preds = loaded_model.predict(test_df)

    # Now output to final location
    # TODO output preds as two columns to location


if __name__ == '__main__':
    proj_dir = os.path.abspath(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), '../'))
    input_filename = os.path.join(proj_dir, 'data/raw/test.csv')
    model_filename = os.path.join(proj_dir, 'models/random_forest_v1.pkl')
    output_filename = os.path.join(proj_dir, 'data/processed/test_set_evaluation.csv')
    print('Reading input data from {}'.format(input_filename))
    print('Reading Model from {}'.format(model_filename))
    print('Writing evaluated data into file {}'.format(output_filename))
    main(input_filename, model_filename, output_filename)
