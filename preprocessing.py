import pandas as pd
import numpy as np

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder
from scipy import stats
from scipy.stats import skew

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
    featureScores = pd.concat([dfcolumns,dfscores], axis=1)
    featureScores.columns = ['Feature','Score']
    print(featureScores.nlargest(20,'Score'))

"""
    Desc: Returns the Z-score values that exceed the set threshold
    Param1: Dataset
    Param2: Feature
    Param3: Threshold
    Output: Fetures exceeding the Z-score according to the threshold
"""
def z_score(df, feat, threshold):
    z = np.abs(stats.zscore(df[feat]))
    return (np.where(z > threshold))[0]

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

"""
    Desc: Gets all features with a numeric data type
    Param1: Dataset
    Output: All features with numeric data type
"""
def get_all_numerical(df):
    # Finding numeric features
    numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numeric = []
    for i in df.columns:
        if df[i].dtype in numeric_dtypes:
            numeric.append(i)

    return numeric

"""
    Desc: Gets all features with a categorical data type
    Param1: Dataset
    Output: All features with categorical data type
"""
def get_all_categorical(df):
    cols = df.columns
    num_cols = get_all_numerical(df)
    categorical = list(set(cols) - set(num_cols))

    return categorical

"""
    Desc: Combines all features from the training and test dataset in 
          preparation for data transformation
    Param1: Training Dataset
    Param2: Test Dataset
    Param3: Target feature to be removed during processing
    Output: Training and tesst dataset merged together
"""
def combine_train_test(train, test, target):
    train_labels = train[target].reset_index(drop=True)
    train_features = train.drop([target], axis=1)
    test_features = test
    all_features = pd.concat([train_features, test_features]).reset_index(drop=True)

    return all_features

"""
    Desc: Retrieves the percent missing values in every feature
    Param1: Dataset
    Output: Dictionary of the percentage missing values of every feature
"""
def percent_missing(df):
    data = pd.DataFrame(df)
    df_cols = list(pd.DataFrame(data))
    dict_x = {}
    for i in range(0, len(df_cols)):
        dict_x.update({df_cols[i]: round(data[df_cols[i]].isnull().mean()*100,2)})
    
    return sorted(dict_x.items(), key=lambda x: x[1], reverse=True)

def fill_numeric_missing(features):
    numeric = get_all_numerical(features)

    features.update(features[numeric].fillna(0))  

    return features

# Log transformation
def logs(res, ls):
    m = res.shape[1]
    for l in ls:
        res = res.assign(newcol=pd.Series(np.log(1.01+res[l])).values)   
        res.columns.values[m] = l + '_log'
        m += 1

    return res

# Square transformation
def squares(res, ls):
    m = res.shape[1]
    for l in ls:
        res = res.assign(newcol=pd.Series(res[l]*res[l]).values)   
        res.columns.values[m] = l + '_sq'
        m += 1

    return res 

# https://stackoverflow.com/questions/17778394/list-highest-correlation-pairs-from-a-large-correlation-matrix-in-pandas
def get_redundant_pairs(df):
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))

    return pairs_to_drop

# Keep in mind that this only works on numeric values
# https://stackoverflow.com/questions/17778394/list-highest-correlation-pairs-from-a-large-correlation-matrix-in-pandas
def get_top_abs_correlations(df, n=10):
    df_numeric = df[get_all_numerical(df)]
    au_corr = df_numeric.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df_numeric)
    au_corr = au_corr.drop(labels=labels_to_drop)
    au_corr_sort = au_corr.sort_values(ascending=False)
    return au_corr_sort[0:n]

def get_top_abs_correlation_to_target(df, target, n=10):
    df_numeric = df[get_all_numerical(df)]
    au_corr = df_numeric.corrwith(df_numeric[target]).drop(target)
    au_corr_sort = au_corr.sort_values(ascending=False)
    return au_corr_sort[0:n]

def get_duplicates(df):
    unique = len(set(df.Id))
    tot_ids = df.shape[0]
    duplicate_ids = tot_ids - unique
    return duplicate_ids

def drop_id_col(df):
    df.drop("Id", axis = 1, inplace = True)
    return df

def get_skewed_feats(df, threshold):
    numeric = get_all_numerical(df)
    skewed_feats = df[numeric].apply(lambda x: skew(x.dropna()))
    skewed_feats = skewed_feats[skewed_feats > threshold]
    skewed_feats = skewed_feats.index
    return skewed_feats