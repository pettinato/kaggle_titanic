
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def build_features(raw_df):
    """
    From Training Set build features,
      Pclass is transformed to an integer
      SibSp is transformed to an integer
      Fare is transformed to an integer
      Sex is transformed to 1/0 with 1 as male
      Age is normalized and nans are filled with the mean Age
      Parch is dummified
      Cabin is transformed into multiple columns as below.
      Embarked is dummified
      Survived is unchanged
    :param df: Data Frame with the above columns
    :return: New DataFrame with features as above
    """
    features_df = pd.DataFrame.copy(raw_df)

    features_df.Pclass = normalize(features_df.Pclass)
    features_df.SibSp = normalize(features_df.SibSp)
    features_df.Fare = normalize(features_df.Fare)

    # # Transform Cabin
    # for cabin_letter in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T']:
    #     features_df['Cabin_' + cabin_letter] = features_df.Cabin.str.contains(cabin_letter).fillna(False).astype(int)
    #
    # features_df['Cabin_Number'] = features_df.Cabin.str.extract('[A-Z]([0-9]*)', expand=False)
    # features_df.Cabin_Number = features_df.Cabin_Number.replace('', np.nan)
    #
    # # Fill the NaN Cabin Numbers with the max Cabin Number.  Assumes that lower cabin numbers are better.
    # features_df['Cabin_Number_NaN'] = features_df.Cabin_Number.isnull().astype(int)
    # features_df.loc[features_df.Cabin_Number.isnull(), ('Cabin_Number')] = features_df.Cabin_Number.max()
    #
    # # Fill the missing Cabin_Numbers with the Cabin_Number mean
    # features_df.Cabin_Number = features_df.Cabin_Number.fillna(features_df.Cabin_Number.mean())
    # features_df.Cabin_Number = features_df.Cabin_Number.astype(int)
    #
    # # Transform Sex
    # features_df['Sex'] = (features_df['Sex'] == 'male').astype(int)
    #
    # # Transform Embarked
    # features_df = pd.get_dummies(features_df, columns=['Embarked'], dummy_na=True)
    #
    # # Age is sometimes null, so replace the null values with the mean
    # features_df['Age'] = features_df.Age.fillna(features_df.Age.mean())
    #
    # features_df = features_df.drop(labels=['Cabin'], axis=1)
    #
    # assert 0 == features_df.isnull().sum().sum(), 'All cells should be non-nan'

    return features_df
