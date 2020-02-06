import pandas as pd
import tensorflow as tf
import numpy as np


from categorical_mappings import *
from feature_bins import *
from preprocessing import *
from plotting import *

DEBUG = 0
DROPPED_COLUMNS = ['OverallCond', 'MSZoning', 'LotFrontage', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Condition1', 'Condition2', 'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType']


def get_avg_sale_given_column(df, column, count=False):
    avg = df.groupby(column, as_index=False)['SalePrice'].mean().sort_values(by='SalePrice')
    avg = avg.rename(columns={'SalePrice': 'AvgSalePrice'})
    if(count):
        counts = df[column].value_counts().values
        avg['Count'] = counts.reshape(-1, 1)
    
    return avg


def compute_histogram_bins(df, bin_size):
    min_val = np.min(df)
    max_val = np.max(df)
    min_boundary = -1.0 * (min_val % bin_size - min_val)
    max_boundary = max_val - max_val % bin_size + bin_size
    n_bins = int((max_boundary - min_boundary) / bin_size) + 1
    bins = np.linspace(min_boundary, max_boundary, n_bins)
    
    return bins

def config_columns(df):
    df = bin_LotArea(df)
    df = bin_YearBuilt(df)
    df = bin_GarageType(df)
    df = has_pool(df)
    df['Neighborhood'] = df['Neighborhood'].map(NEIGHBORHOOD_MAPPING)
    df['BldgType']   = df['BldgType'].map(BLDGTYPE_MAPPING)
    df['HouseStyle'] = df['HouseStyle'].map(HOUSESTYLE_MAPPING)
    df['GarageType'] = df['GarageType'].map(GARAGE_MAPPING)
    df['GarageType'] = df['GarageType'].fillna(0)
    df['SaleCondition'] = df['SaleCondition'].map(SALECONDITION_MAPPING)
    
    return df

def prune_features(train, test):
    #enc_train = str_encode_to_int(train)
    #enc_train = encode_label(train, ['SaleCondition, GarageType'])
    cat_columns = convert_object_to_category(train)
    print(train.head())
    #print(cat_columns)
    labels = get_classes(cat_columns)
    #print(labels)
    label_dicts = cateogrical_to_numerical(train, labels)
    #print(label_dicts)
    enc_train = map_categorical_to_numerical(train, label_dicts)
    print(enc_train.head())
    univariate_selection(enc_train)
    #print(filter_cat_cols(train).info())
    #print(get_col_null_sum(train))
    train = drop_columns(train, DROPPED_COLUMNS)
    test = drop_columns(test, DROPPED_COLUMNS)

    train = config_columns(train)
    test = config_columns(test)


    #plot_freq_dist(train, 'HouseStyle')

    if(DEBUG):
        curr_column = 'SaleCondition'
        tt = get_avg_sale_given_column(train, curr_column, True)
        plt.bar(tt[curr_column], tt['AvgSalePrice'], align='center')
        plt.xlabel('Value')
        plt.ylabel('Counts')
        plt.grid(True)
        plt.show()

        print(train.shape, test.shape)
        print("Missing Values: ", get_null_vals(train[curr_column]))
        print(train[curr_column].describe())
        print(train.head())
        print(get_avg_sale_given_column(train, curr_column, True))
        print(train[curr_column].value_counts())
        #train.boxplot(curr_column,'SalePrice',rot = 30,figsize=(5,6))

    return train, test
    #print(train["Neighborhood"].value_counts())

if __name__ == "__main__":
    train = pd.read_csv("/Users/Kukus/Desktop/House_Prices_Kaggle/Data/train.csv")
    test = pd.read_csv("/Users/Kukus/Desktop/House_Prices_Kaggle/Data/test.csv")
    train, test = prune_features(train, test)

