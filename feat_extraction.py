import pandas as pd

from categorical_mappings import *
from feature_bins import *
from plotting import *
from preprocessing import *

def process_features(train, test):
    cat_columns = convert_object_to_category(train)
    labels      = get_classes(cat_columns)
    label_dicts = cateogrical_to_numerical(train, labels)
    enc_train   = map_categorical_to_numerical(train, label_dicts)
    
    return enc_train


def outlier_processing(df):
    
    df = df.drop(z_score(df, 'GrLivArea', 6))

    return df


def refactor_features(df):
    df = df.drop(['Utilities', 'Street', 'PoolQC',], axis=1)
    
    # Merging and Feature Creation
    df['YearsSinceRemodel']     = df['YrSold'].astype(int) - df['YearRemodAdd'].astype(int)
    df['Total_Home_Quality']    = df['OverallQual'] + df['OverallCond']
    df['TotalSF']               = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']

    df['Total_sqr_footage'] = (df['BsmtFinSF1'] + df['BsmtFinSF2'] + df['1stFlrSF'] + df['2ndFlrSF'])
    df['MasVnrArea']        = df['MasVnrArea'].apply(lambda x: np.exp(4) if x <= 0.0 else x)
    df['BsmtFinSF1']        = df['BsmtFinSF1'].apply(lambda x: np.exp(6.5) if x <= 0.0 else x)
    df['Total_Bathrooms']   = (df['FullBath'] + (0.5 * df['HalfBath']) + df['BsmtFullBath'] + (0.5 * df['BsmtHalfBath']))
    df['TotalBsmtSF']       = df['TotalBsmtSF'].apply(lambda x: np.exp(6) if x <= 0.0 else x)
    
    # Booleans
    df['HasFireplace']      = df['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
    df['HasPool']           = df['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
    df['HasGarage']         = df['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
    df['Has2ndFloor']       = df['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
    df['HasBsmt']           = df['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

    return df

if __name__ == "__main__":

    df_train = pd.read_csv("Data/train.csv")
    df_test = pd.read_csv("Data/test.csv")
    var = 'TotalBsmtSF'

    # Remove the IDs
    train_ID = df_train['Id']
    test_ID = df_test['Id']
    df_train.drop(['Id'], axis=1, inplace=True)
    df_test.drop(['Id'], axis=1, inplace=True)

    all_features = combine_train_test(df_train, df_test, 'SalePrice')
    
    missing = percent_missing(all_features)
    all_features = fill_numeric_missing(all_features)
    df_train = refactor_features(df_train)

    # in case of positive skewness, log transformations usually works well.
    df_train['SalePrice'] = np.log(df_train['SalePrice'])

    # Process chosen features
    df_train = outlier_processing(df_train)
    df_train['GrLivArea'] = np.log(df_train['GrLivArea'])
    df_train.loc[df_train['HasBsmt']==1,'TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])

    # Convert categorical variable into dummy
    df_train = pd.get_dummies(df_train)
    print(get_skewed_feats(df_train, 0.5))