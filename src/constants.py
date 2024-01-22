delete_col_list =  ['index','zonedist2', 'zonedist3', 'zonedist2', 'zonedist3', 'zonedist4', 
                    'overlay1', 'overlay2', 'spdist1', 'spdist2', 'spdist3', 'ltdheight', 'ext',
                    'histdist', 'landmark']

categorical_cols = ['borough', 'block', 'lot', 'schooldist', 'council', 'zipcode', 'firecomp', 'policeprct',
                    'healthcenterdistrict', 'healtharea', 'sanitboro', 'sanitdistrict', 'sanitsub',
                    'zonedist1', 'splitzone', 'landuse', 'proxcode','irrlotcode', 'lottype', 'bsmtcode',
                    'zonemap']

model_cat_cols = ['borough', 'splitzone', 'irrlotcode']

numeric_cols = ['easements', 'lotarea', 'bldgarea', 'comarea', 'resarea', 'officearea', 'retailarea', 'garagearea', 
                'strgearea', 'factryarea', 'otherarea', 'numbldgs', 'numfloors', 'unitstotal', 'lotfront', 'lotdepth', 
                'bldgfront', 'bldgdepth', 'assessland', 'assesstot', 'exemptland', 'exempttot', 'yearbuilt', 'yearalter1',
                 'yearalter2', 'builtfar', 'tract2010', 'xcoord', 'ycoord']

# Define a logistic regression hyperparameter grid to search over
logistic_param_grid = {
    'classifier__C': [0.001, 0.01, 0.1, 1],
    'classifier__penalty': ['l1', 'l2', 'elasticnet']
    }

# Define hyperparameter grid for the Random Forest
randomforest_param_grid = {
    'classifier__n_estimators': [10, 15, 20],  # Number of trees in the forest
    'classifier__max_depth': [5, 10, 15],  # Maximum depth of the tree
    'classifier__min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'classifier__min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
    'classifier__max_features': ['auto', 'sqrt', 'log2'],  # Number of features to consider for the best split
    'classifier__bootstrap': [True, False],  # Whether bootstrap samples are used when building trees
    'classifier__criterion': ['gini', 'entropy']  # Split criterion
}

# Define hyperparameter grid for the Decision Tree
decisiontree_param_grid = {
    'classifier__criterion': ['gini', 'entropy'],
    'classifier__max_depth': [1,2,3],
    'classifier__min_samples_split': [5, 10, 15],
    'classifier__min_samples_leaf': [5, 10, 15],
    'classifier__max_features': ['auto']
    }
