import pandas as pd
import numpy as np
from pprintpp import pprint
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from constants import categorical_cols, numeric_cols, logistic_param_grid, decisiontree_param_grid, randomforest_param_grid

def get_typecasted_cols(train, test):
    """
    The function `get_typecasted_cols` takes in two dataframes, `train` and `test`, and typecasts the
    columns specified in `categorical_cols` to the 'category' data type and the columns specified in
    `numeric_cols` to the 'float' data type, and then returns the modified `train` and `test`
    dataframes.
    
    :param train: The `train` parameter is a DataFrame containing the training data
    :param test: The "test" parameter is a DataFrame containing the test data
    :return: the modified train and test dataframes with the specified columns typecasted.
    """
    train[categorical_cols] = train[categorical_cols].astype('category')
    train[numeric_cols] = train[numeric_cols].astype(float)

    test[categorical_cols] = test[categorical_cols].astype('category')
    test[numeric_cols] = test[numeric_cols].astype(float)

    return train, test


def remove_highly_correlated_features(df, threshold=0.8):
    """
    The function `remove_highly_correlated_features` takes a DataFrame as input and removes columns that
    have a correlation greater than a specified threshold.
    
    :param df: The parameter `df` is a pandas DataFrame that contains the dataset with the features
    :param threshold: The threshold parameter is used to determine the level of correlation between
    features that is considered "highly correlated". Any pair of features with a correlation coefficient
    greater than the threshold value will be considered highly correlated and one of them will be
    dropped
    :return: The function `remove_highly_correlated_features` returns two values: `df_filtered` and
    `to_drop`.
    """
    # Calculate the correlation matrix
    corr_matrix = df.corr().abs()

    # Select upper triangle of correlation matrix
    upper_triangle = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    upper_matrix = corr_matrix.where(upper_triangle)

    # Find index of feature columns with correlation greater than the threshold
    to_drop = [column for column in upper_matrix.columns if any(upper_matrix[column] > threshold)]

    print(f'Dropping following columns')
    pprint(to_drop)

    # Drop highly correlated features
    df_filtered = df.drop(columns=to_drop)

    return df_filtered, to_drop


def get_log_features(df, numeric_cols):
    """
    The function `get_log_features` performs a log transformation on right-skewed numeric columns in a
    dataframe and returns a new dataframe with the transformed columns added.
    
    :param df: The parameter `df` is a pandas DataFrame that contains the data you want to perform log
    transformation on
    :param numeric_cols: The `numeric_cols` parameter is a list of column names in the dataframe `df`
    that contain numeric data. These columns will be selected for log transformation
    :return: a new dataframe `df_log` that contains the original dataframe `df` with log-transformed
    numeric columns added.
    """
    # Perform log transformation on right-skewed numeric columns
    df_log_transformed = df[numeric_cols].apply(lambda x: np.log1p(x))
    df_log_transformed.drop(['xcoord', 'ycoord'], inplace = True, axis = 1)

    # Add a suffix to the new columns to differentiate them
    suffix = '_log_transformed'
    df_log_transformed.columns = [col + suffix for col in df_log_transformed.columns]

    # Drop the original numeric columns from the original dataframe
    df_log = df.drop(numeric_cols, axis=1)

    # Concatenate the original dataframe with the log-transformed numeric columns
    df_log = pd.concat([df_log, df_log_transformed], axis=1)

    return df_log


def getLogisticRegressionResults(data_dict, categorical_columns, numerical_columns):
    """
    The function `getLogisticRegressionResults` performs logistic regression on a given dataset,
    including preprocessing steps such as one-hot encoding and scaling, and returns the best estimator,
    feature names, and predicted probabilities.
    
    :param data_dict: The `data_dict` parameter is a dictionary that contains the training and testing
    data. It should have the following keys:
    :param categorical_columns: The `categorical_columns` parameter is a list of column names or indices
    that represent the categorical features in your dataset. These features will be one-hot encoded
    during preprocessing
    :param numerical_columns: The `numerical_columns` parameter is a list of column names that represent
    the numerical features in your dataset. 
    :return: three values: 
    1. `best_estimator`: The best estimator found by the GridSearchCV, which includes the preprocessor
    and logistic regression classifier.
    2. `feature_names`: A list of feature names after one-hot encoding and scaling.
    3. `y_prob`: The predicted probabilities for the positive class (class 1) for each instance in the
    test set.
    """
    X_train, y_train = data_dict['x_train'], data_dict['y_train']
    X_test, y_test = data_dict['x_test'], data_dict['y_test']
    
    # Create a ColumnTransformer to apply one-hot encoding to categorical features and scale numerical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), categorical_columns),
            ('num', StandardScaler(), numerical_columns)
        ], 
        remainder='passthrough'  # Include any other non-specified columns as-is
    )

    # Create a Pipeline with preprocessing and Logistic Regression
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(class_weight='balanced'))
    ])

    # Initialize Stratified 5-Fold Cross-Validation
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scorer = make_scorer(f1_score)

    # Initialize GridSearchCV for hyperparameter tuning with F1 scoring
    grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=logistic_param_grid,
    cv=kfold,
    scoring=scorer  # Use F1 score for hyperparameter tuning
    )

    # Fit the GridSearchCV on the data to find the best hyperparameters
    grid_search.fit(X_train, y_train)

    # Print the best hyperparameters
    best_params = grid_search.best_params_
    print("Best Hyperparameters:")
    pprint(best_params)


    best_estimator = grid_search.best_estimator_


    # Make predictions on the test set
    y_pred = grid_search.predict(X_test)
    y_prob = grid_search.predict_proba(X_test)[:, 1]

    # Compute the evaluation scores
    model_f1 = f1_score(y_test, y_pred)
    print(f"Test F1 score : {model_f1}")
    
    # Extract the feature names after one-hot encoding
    preprocessor = best_estimator.named_steps['preprocessor']
    feature_names = preprocessor.transformers_[0][1].get_feature_names(input_features=categorical_columns).tolist() + numerical_columns

    
    return best_estimator, feature_names, y_prob



def getDecisionTreeResults(data_dict, categorical_columns):
    """
    The function `getDecisionTreeResults` trains a decision tree classifier using the provided training
    data and performs hyperparameter tuning using grid search with cross-validation, and returns the
    best estimator and preprocessor.
    
    :param data_dict: The `data_dict` parameter is a dictionary that contains the training and testing
    data. It should have the following keys:
    :param categorical_columns: The `categorical_columns` parameter is a list of column names or indices
    that represent the categorical features in your dataset. These features will be one-hot encoded
    before being used in the decision tree model
    :return: the best estimator (trained decision tree model) and the preprocessor used for data
    transformation.
    """
    X_train, y_train = data_dict['x_train'], data_dict['y_train']
    X_test, y_test = data_dict['x_test'], data_dict['y_test']
    
    # Create a ColumnTransformer to apply one-hot encoding to categorical features and scale numerical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), categorical_columns)
        ], 
        remainder='passthrough'  # Include any other non-specified columns as-is
    )
    
    # Create a Pipeline for Decision Tree
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', DecisionTreeClassifier())
    ])

    # Initialize Stratified 5-Fold Cross-Validation
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scorer = make_scorer(f1_score)

    # Initialize GridSearchCV for hyperparameter tuning with F1 scoring
    grid_search = GridSearchCV(estimator=pipeline, param_grid=decisiontree_param_grid, cv=kfold, scoring=scorer)

    # Fit the GridSearchCV on the training data
    grid_search.fit(X_train, y_train)

    # Print the best hyperparameters
    best_params = grid_search.best_params_
    print("Best Hyperparameters:")
    pprint(best_params)

    # Make predictions on the test set
    y_pred = grid_search.predict(X_test)

    # Compute the evaluation scores
    model_f1 = f1_score(y_test, y_pred)
    print(f"Test F1 score : {model_f1}")

    # Get the best estimator
    best_estimator = grid_search.best_estimator_
    
    return best_estimator, preprocessor


def getRandomForestResults(data_dict, categorical_columns):
    """
    The function `getRandomForestResults` trains a Random Forest classifier using the provided data and
    categorical columns, performs hyperparameter tuning using randomized search, and returns the best
    estimator and preprocessor.
    
    :param data_dict: The `data_dict` parameter is a dictionary that contains the training and testing
    data. It should have the following keys:
    :param categorical_columns: The `categorical_columns` parameter is a list of column names or indices
    that represent the categorical features in your dataset. These features will be one-hot encoded
    before being used in the Random Forest model
    :return: the best estimator and the preprocessor.
    """
    X_train, y_train = data_dict['x_train'], data_dict['y_train']
    X_test, y_test = data_dict['x_test'], data_dict['y_test']

    # Create a ColumnTransformer to apply one-hot encoding to categorical features and scale numerical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), categorical_columns)
        ], 
        remainder='passthrough'  # Include any other non-specified columns as-is
    )

    # Create a Pipeline for Random Forest
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier())
    ])

    # Initialize Stratified 5-Fold Cross-Validation
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scorer = make_scorer(f1_score)

    random_search = RandomizedSearchCV(estimator=pipeline, param_distributions=randomforest_param_grid, cv=kfold, scoring=scorer, n_jobs=-1, n_iter=100)


    # Fit the GridSearchCV on the training data
    random_search.fit(X_train, y_train)

    # Print the best hyperparameters
    best_params = random_search.best_params_
    print("Best Hyperparameters:")
    pprint(best_params)

    # Make predictions on the test set
    y_pred = random_search.predict(X_test)

    # Compute the evaluation scores
    model_f1 = f1_score(y_test, y_pred)
    print(f"Test F1 score : {model_f1}")
    # Get the best estimator
    best_estimator = random_search.best_estimator_

    return best_estimator, preprocessor