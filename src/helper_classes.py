from pprintpp import pprint


class MissingValueImputer:
    '''
    The `MissingValueImputer` class is a Python class that can be used to fit and transform a pandas
    DataFrame by imputing missing values with either the mean (for numerical columns) or the mode (for
    categorical columns).
    '''
    def __init__(self):
        """
        The function initializes an empty dictionary called "imputation_values" as an attribute of the
        class.
        """
        self.imputation_values = {}

    def fit(self, df):
        """
        The `fit` function calculates and stores the mean for numerical columns and the mode for categorical
        columns in a dictionary called `imputation_values`.
        
        :param df: The parameter `df` is a pandas DataFrame that contains the data you want to fit the
        imputer on. It is assumed that the DataFrame has columns with different data types, such as
        numerical (int64, float64) and categorical. The code iterates over each column in the DataFrame and
        checks
        """
        for column in df.columns:
            if df[column].dtype in ['int64', 'float64']:
                # Store mean for numerical columns
                self.imputation_values[column] = df[column].mean()
            else:
                # Store mode for categorical columns
                self.imputation_values[column] = df[column].mode()[0]
        print('Following are the imputation values : ')
        pprint(self.imputation_values)

    def transform(self, df):
        """
        The function fills missing values in a DataFrame using stored imputation values.
        
        :param df: The parameter `df` is a pandas DataFrame object that represents the dataset you want to
        transform
        :return: the transformed dataframe after imputing missing values.
        """
        for column in df.columns:
            # Impute missing values based on stored values
            df[column].fillna(self.imputation_values[column], inplace=True)

        return df