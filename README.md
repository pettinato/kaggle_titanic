kaggle_titanic
==============================

A run through of the Kaggle Titanic model as per https://www.kaggle.com/c/titanic

Used as an example to structure analysis and modeling.

Analysis Steps
------------
1. notebooks/initial_analysis.ipynb
   * Used to do initial analysis on the raw data
2. scripts/make_featureset.py 
   * Used to build the featureset from the raw data
3. TODO

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── scripts            <- Scripts to run to generate features, models, etc
    │ 
    ├── src                <- Source code for use in this project as a Python module
    │   │
    │   ├── features       <- libraries to turn raw data into features for modeling
    │   │
    │   ├── models         <- libraries to train models and then use trained models to make
    │   │   │                 predictions
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py


--------

<p><small>Project partially based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
