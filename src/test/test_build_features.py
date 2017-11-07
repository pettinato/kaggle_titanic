

import pandas as pd
import numpy as np
import src.features.build_features as bf


def compare_dataframes(exp_df, act_df):
    """
    Compare two dataframes ignoring row order
    :param exp_df:
    :param act_df:
    """
    def ah(exp, act, prefix):
        assert exp == act, '{prefix}\nExpected {exp}\nFound {act}' \
            .format(prefix=prefix, exp=exp, act=act)
    e_df = pd.DataFrame.copy(exp_df)
    a_df = pd.DataFrame.copy(act_df)
    ah(set(e_df.columns), set(a_df.columns), 'Columns')
    ah(e_df.shape, a_df.shape, 'Shape')

    cols = e_df.columns.tolist()
    e_df = e_df.sort_values(cols)
    a_df = a_df.sort_values(cols)

    for e_tup, a_tup in zip(e_df.iterrows(), a_df.iterrows()):
        erow = e_tup[1]
        arow = a_tup[1]
        for col in cols:
            assert erow[col] == arow[col] or np.isclose(erow[col], arow[col]), \
                'Diff in column {col}\nExpected row\n{exp}\nActual row\n{act}'.format(
                    col=col, exp=erow, act=arow, exprow=erow, actrow=arow
                )


def test_build_features_basic():
    """Test that the features are properly created
        1. Survived (dependent variable)
        2. Pclass - dummify
        3. Sex - 1 if male, 0 if female
        4. Fare - Normalized - Will need to store the mean/stdev
           to a file in order to normalize the test set.
        5. Cabin - as 5 columns.  1 if nan, 1 each for if contains B, C, D, E
        6. Embarked - dummify without nan dummy column
    """
    df = pd.DataFrame(dict(
        Survived=['A', 'A', 'B'],
        Pclass=[1, 2, 2],
        Sex=['male', 'male', 'female'],
        Fare=[27.4, 18.2, 10],
        Cabin=['B12', 'A32', 'D12'],
        Embarked=['A', 'B', 'A'],
        OtherStuff=[1, 2, 3]  # Should be dropped
    ))

    result = bf.build_features(df)

    expected = pd.DataFrame(dict(
        Survived=['A', 'A', 'B'],
        Pclass_1=[1, 0, 0],
        Pclass_2=[0, 1, 1],
        Sex=[1, 1, 0],
        Fare=[27.4, 18.2, 10],
        Cabin_B=[1, 0, 0],
        Cabin_C=[0, 0, 0],
        Cabin_D=[0, 0, 1],
        Cabin_E=[0, 0, 0],
        Cabin_nan=[0, 0, 0],
        Embarked_A=[1, 0, 1],
        Embarked_B=[0, 1, 0]
    ))

    compare_dataframes(expected, result)


def test_build_features_cabin():
    """Test that the features are properly created
        1. Survived (dependent variable)
        2. Pclass - dummify
        3. Sex - 1 if male, 0 if female
        4. Fare - Normalized - Will need to store the mean/stdev
           to a file in order to normalize the test set.
        5. Cabin - as 5 columns.  1 if nan, 1 each for if contains B, C, D, E
        6. Embarked - dummify without nan dummy column
    """
    df = pd.DataFrame(dict(
        Survived=['A', 'B']*3,
        Pclass=[0]*6,
        Sex=['male', 'female']*3,
        Fare=[27.4, 20.5]*3,
        Cabin=['B12 A32 C12', np.nan, 'B12', 'C13', 'E14 D14', 'C5 E12 D423 B12'],
        Embarked=['A', 'B']*3,
    ))

    result = bf.build_features(df)

    expected = pd.DataFrame(dict(
        Survived=['A', 'B'] * 3,
        Pclass_0=[1]*6,
        Sex=[1, 0]*3,
        Fare=[27.4, 20.5] * 3,
        Cabin_B=[1, 0, 1, 0, 0, 1],
        Cabin_C=[1, 0, 0, 1, 0, 1],
        Cabin_D=[0, 0, 0, 0, 1, 1],
        Cabin_E=[0, 0, 0, 0, 1, 1],
        Cabin_nan=[0, 1, 0, 0, 0, 0],
        Embarked_A=[1, 0]*3,
        Embarked_B=[0, 1]*3
    ))
    compare_dataframes(expected, result)
