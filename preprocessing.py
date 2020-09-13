import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder

def univariate_selection(df):
    X = df.iloc[:,0:-1]
    X = X.fillna(0)

    Y = df.iloc[:,-1]
    Y = Y.values.reshape(-1, 1)
    
    #Apply SelectKBest class to extract 10 best features
    bestfeatures = SelectKBest(score_func=chi2, k=10)
    fit = bestfeatures.fit(X, Y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)

    #concat two dataframes for better visualization 
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Feature','Score']
    print(featureScores.nlargest(40,'Score'))


"""
    Desc: Encodes categorical values to numerical
    Param1: Dataset
    Output: Categorical features encoded to integer, ranging from 0 upwards
"""
def str_encode_to_int(df):
    # Temporary Solution
    df = df.fillna(0)
    print(df.info())

    le = LabelEncoder()
    le.fit(df)
    df_enc = le.transform(df)
    return df_enc

"""
    Desc: Encodes label of CATEGORY type
    Param1: Dataset
    Param2: Column
    Output: Features of object type 
"""
def encode_label(df, column):
    df[column] = df[column].cat.codes
    return df

"""
    Desc: Returns features of object type
    Param1: Dataset
    Output: Features of object type 
"""
def filter_cat_cols(df):
    df = df.select_dtypes(include=['object'])
    return df

"""
    Desc: Returns features converted from object type to category
    Param1: Dataset
    Output: Features of category type 
"""
def convert_object_to_category(df):
    df_objects = df.select_dtypes(include=['object'])
    df = df_objects.astype('category')
    return df
    
"""
    Desc: Gets the total amount of null values
    Param1: Dataset
    Output: Total number of null values
"""
def get_null_sum(df):
    return df.isnull().values.sum()

"""
    Desc: Get the amount of null values in each column
    Param1: Dataset
    Output: list of null values for each column
"""
def get_col_null_sum(df):
    return df.isnull().sum()

"""
    Desc: Drops selected columns from dataset
    Param1: Dataset
    Param2: Array of column names to be dropped
    Output: Dataset with selected columns dropped
"""
def drop_columns(df, columns):
    df = df.drop(columns, axis=1)
    return df

"""
    Desc: Returns classnames of the dataset
    Param1: Dataset
    Output: List of column names/features
"""
def get_classes(df):
    return list(df.columns.values)

"""
    Desc: Returns the frequency distribution of the columns values
    Param1: Dataset
    Param2: Column name
    Output: Frequency distribution of the input dataset and column name
"""
def freq_dist(df, column):
    return df[column].value_counts()

"""
    Desc: Applies a numerical number for each categorical value
    Param1: Dataset
    Param2: Column name
    Output: Dictionary of the mapping from cateogrical to numerical
"""
def cateogrical_to_numerical(df, columns):
    replace_map_comp = dict()
    for i in columns:
        labels = df[i].astype('category').cat.categories.tolist()
        mapped_column = {i : {k: v for k, v in zip(labels, list(range(1,len(labels)+1)))}}
        replace_map_comp[i] = mapped_column[i]
    return replace_map_comp

"""
    Desc: Replaces existing cateogrical value with set numerical value
    Param1: Dataset
    Param2: Mapping Dictionary
    Output: Updated dataset with numerical mapping implemented
"""
def map_categorical_to_numerical(df, mapping):
    for k, v in mapping.items():
        #df[k] = df[k].replace(v, inplace=True)
        df[k] = df[k].replace(v)
    return df