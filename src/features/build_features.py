
import pandas as pd


def read_raw_data(filename):
    """
    Read in the raw dataframe and drop columns,
        PassengerId
        Name
        Age
        SibSp
        Parch
        Ticket
    Expects a csv file with a header.
    :param filename: A filename to read data from.
    :return: DataFrame
    """
    drop_cols = ['PassengerId', 'Name', 'Age', 'SibSp', 'Parch', 'Ticket']
    df = pd.read_csv(filename)
    return df.drop(labels=drop_cols, axis=1)


def build_features(raw_df):
    """
    From Training Set build features,
        1. Survived (dependent variable)
        2. Pclass - dummify
        3. Sex - 1 if male, 0 if female
        4. Fare - not transformed
        5. Cabin - as 5 columns.  1 if nan, 1 each for if contains B, C, D, E
        6. Embarked - dummify without nan dummy column
    :param df: Data Frame with the above columns
    :return: New DataFrame with features as above and all other columns dropped
    All columns returned will be numeric data types.
    """
    req_cols = ['Survived', 'Pclass', 'Sex', 'Fare', 'Cabin', 'Embarked']
    features_df = pd.DataFrame.copy(raw_df[req_cols])
    features_df = pd.get_dummies(features_df, columns=['Pclass', 'Embarked'])
    features_df['Sex'] = (features_df['Sex'] == 'male').astype(int)
    features_df['Cabin_nan'] = features_df.Cabin.isnull().astype(int)

    for cabin in ['B', 'C', 'D', 'E']:
        features_df['Cabin_' + cabin] = \
            features_df.Cabin.fillna('').str.contains(cabin).astype(int)
    features_df = features_df.drop(labels=['Cabin'], axis=1)
    return features_df
