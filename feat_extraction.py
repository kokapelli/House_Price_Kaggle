import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from categorical_mappings import *

DROPPED_COLUMNS = ['MSZoning', 'LotFrontage', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Condition1', 'Condition2', 'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType']

def drop_columns(df, columns):
    df = df.drop(columns, axis=1)
    return df

def get_null_vals(df):
    return df.isnull().sum()

def get_classes(df):
    return list(df.columns.values)

def get_avg_sale_given_column(df, column, count=False):
    avg = df.groupby(column, as_index=False)['SalePrice'].mean().sort_values(by='SalePrice')
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

def bin_LotArea(df):
    """
        mean      10516.828082
        std        9981.264932
        min        1300.000000
        25%        7553.500000
        50%        9478.500000
        75%       11601.500000
        max      215245.000000
    """

    df.loc[ df['LotArea'] <= 2000, 'LotArea'] = 0,
    df.loc[(df['LotArea'] > 2000)   & (df['LotArea'] <= 4000),   'LotArea'] = 1,
    df.loc[(df['LotArea'] > 4000)   & (df['LotArea'] <= 6000),   'LotArea'] = 2,
    df.loc[(df['LotArea'] > 6000)   & (df['LotArea'] <= 8000),   'LotArea'] = 3,
    df.loc[(df['LotArea'] > 8000)   & (df['LotArea'] <= 10000),  'LotArea'] = 4,
    df.loc[(df['LotArea'] > 10000)  & (df['LotArea'] <= 25000),  'LotArea'] = 5,
    df.loc[(df['LotArea'] > 25000)  & (df['LotArea'] <= 50000),  'LotArea'] = 6,
    df.loc[(df['LotArea'] > 50000)  & (df['LotArea'] <= 100000), 'LotArea'] = 7,
    df.loc[ df['LotArea'] > 100000, 'LotArea'] = 8

    return df

def config_columns(df):
    df = bin_LotArea(df)
    df['Neighborhood'] = df['Neighborhood'].map(NEIGHBORHOOD_MAPPING)

    return df

if __name__ == "__main__":
    train = pd.read_csv("/Users/Kukus/Desktop/House_Prices_Kaggle/Data/train.csv")
    test = pd.read_csv("/Users/Kukus/Desktop/House_Prices_Kaggle/Data/test.csv")


    train = config_columns(train)
    test = config_columns(test)

    neigh = get_avg_sale_given_column(train, 'Neighborhood', True)

    train = drop_columns(train, DROPPED_COLUMNS)
    test = drop_columns(test, DROPPED_COLUMNS)
    
    """
    plt.bar(neigh['Neighborhood'], neigh['SalePrice'], align='center')
    plt.xlabel('Value')
    plt.ylabel('Counts')
    plt.grid(True)
    plt.show()
    """
    
    print(train.shape, test.shape)
    #print(get_avg_sale_given_column(train, 'Neighborhood', True))
    print(get_null_vals(train['Neighborhood']))
    print(train['Neighborhood'].describe())
    print(train.head())
    #print(train["Neighborhood"].value_counts())
