

import pandas as pd
import numpy as np
import src.features.build_features as bf


def test_build_features_numeric():
    """Test that the following features are
    transformed to an integer,
    Pclass, SibSp, Fare"""
    df = pd.DataFrame(dict(
        Pclass=[1, 2, 3],
        Sex=[0, 0, 0],
        SibSp=[1, 2, 3],
        Parch=[1, 1, 1],
        Fare=[1, 2, 3],
        Cabin=['1', '1', '1'],
        Embarked=[1, 1, 1],
        Survived=[1, 1, 1]
    ))

    result = bf.build_features(df)

    # Result from sklearn.preprocessing.normalize
    exp = [0.26726124, 0.53452248, 0.80178373]

    assert all(np.isclose(exp, result.Pclass))
    assert all(np.isclose(exp, result.SibSp))
    assert all(np.isclose(exp, result.Fare))
