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
import pandas as pd
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
    print('Reading in raw data')
    raw_test_df = bf.read_raw_test_data(test_path)
    predicted_df = pd.DataFrame(dict(PassengerId=raw_test_df.PassengerId))
    raw_test_df.drop(labels=['PassengerId'], axis=1, inplace=True)

    # build_features assumes that the input DataFrame has column Survived.
    # As a workaround, add it and then remove it later
    raw_test_df['Survived'] = 0

    # One of the Fare values in this raw data is null.
    # For now, just replace with 0.
    raw_test_df.loc[raw_test_df.Fare.isnull(), 'Fare'] = 0

    print('Building Features')
    test_df = bf.build_features(raw_test_df)
    test_df.drop(labels=['Survived', 'Pclass_2'], axis=1, inplace=True)

    exp_col = {'Sex', 'Fare', 'Pclass_1', 'Pclass_3', 'Embarked_C',
               'Embarked_Q', 'Embarked_S', 'Cabin_nan', 'Cabin_B',
               'Cabin_C', 'Cabin_D', 'Cabin_E'}
    assert exp_col == set(test_df.columns), \
        'Expected columns {0} should match actual columns {1}' \
        .format(exp_col, test_df.columns)

    print('Verifying output model')
    loaded_model = joblib.load(model_path)
    preds = loaded_model.predict(test_df)
    predicted_df['Survived'] = preds

    print('Outputting to final location')
    predicted_df.to_csv(output_path, index=False, columns=['PassengerId', 'Survived'])


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
