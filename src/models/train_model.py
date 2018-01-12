
from sklearn.externals import joblib
import sklearn as sk
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from ..features import build_features


def train_selected_random_forest(train_filename, output_path):
    """
    Load raw training data and output final model.
    :param train_filename: Raw data input location as a csv file with a header.
    :param output_path: Output model location using sklearn.externals.joblib
    """

    raw_train_df = build_features.read_raw_data(train_filename)
    feature_df = build_features.build_features(raw_train_df)

    # Remove the dependent variable, and remove the Pclass_2 as it's highly correlated with other
    # variables as and not a very useful predictor per feature_analysis
    dep_df = feature_df.Survived
    ind_df = feature_df.drop(labels=['Survived', 'Pclass_2'], axis=1)

    forest = make_pipeline(sk.preprocessing.StandardScaler(), RandomForestClassifier())
    best_forest = GridSearchCV(forest,
                               param_grid=dict(randomforestclassifier__n_estimators=[10, 100, 1000],
                                               randomforestclassifier__max_features=list(range(1, 13)),
                                               randomforestclassifier__max_depth=[10, 100, 1000, None]),
                               n_jobs=-1)
    best_forest.fit(ind_df, dep_df)

    print('Best Score: ', best_forest.best_score_)
    best_forest_preds = best_forest.best_estimator_.fit(ind_df, dep_df).predict(ind_df)
    print('F_Score: ', f1_score(best_forest_preds, dep_df))
    print('Precision: ', precision_score(best_forest_preds, dep_df))
    print('Recall: ', recall_score(best_forest_preds, dep_df))
    print('Best Params:\n', '\n'.join(map(str, best_forest.best_params_.items())))

    joblib.dump(best_forest.best_estimator_, output_path)

    # Verify output model
    loaded_model = joblib.load(output_path)
    preds = loaded_model.predict(ind_df)
    match = all(loaded_model.predict(ind_df) == best_forest.best_estimator_.predict(ind_df))
    print('All predictions match expected: ', match)
