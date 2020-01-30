import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


def drop_columns(df, columns):
    df = df.drop(columns, axis=1)
    return df

def get_null_column_sum(df):
    return df.isnull().sum()

def get_classes(df):
    return list(df.columns.values)

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

    return df

if __name__ == "__main__":
    train = pd.read_csv("/Users/Kukus/Desktop/House_Prices_Kaggle/Data/train.csv")
    test = pd.read_csv("/Users/Kukus/Desktop/House_Prices_Kaggle/Data/test.csv")
    print(train.shape, test.shape)

    train = config_columns(train)
    test = config_columns(test)

    """
    columns_to_drop = ['Example']
    drop_columns(train, columns_to_drop)
    drop_columns(test, columns_to_drop)
    """
    
    plt.hist(train['LotArea'])
    plt.xlabel('Value')
    plt.ylabel('Counts')
    plt.title('Compute Bins Example')
    plt.grid(True)
    plt.show()

    #print(train['LotArea'].sort_values())
    print(train['LotArea'].describe())
    print(train["LotArea"].value_counts())

    """
    pclass_pivot = train.pivot_table(index="LotArea",values="SalePrice")
    pclass_pivot.plot.bar()
    plt.show()
    """
    