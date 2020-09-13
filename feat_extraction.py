import pandas as pd
import numpy as np

from plotting import *
from preprocessing import *
from categorical_mappings import *
from feature_bins import *


def process_cols(df):
    df = bin_LotArea(df)
    df = bin_YearBuilt(df)
    df = bin_GarageType(df)
    df = has_pool(df)
    df['Neighborhood']  = df['Neighborhood'].map(NEIGHBORHOOD_MAPPING)
    df['BldgType']      = df['BldgType'].map(BLDGTYPE_MAPPING)
    df['HouseStyle']    = df['HouseStyle'].map(HOUSESTYLE_MAPPING)
    df['GarageType']    = df['GarageType'].map(GARAGE_MAPPING)
    df['GarageType']    = df['GarageType'].fillna(0)
    df['SaleCondition'] = df['SaleCondition'].map(SALECONDITION_MAPPING)
    
    return df

def prune_features(train, test):
    train = process_cols(train)
    test = process_cols(test)

    return train, test

def process_features(train, test):
    cat_columns = convert_object_to_category(train)
    labels      = get_classes(cat_columns)
    label_dicts = cateogrical_to_numerical(train, labels)
    enc_train   = map_categorical_to_numerical(train, label_dicts)
    
    return enc_train

if __name__ == "__main__":
    df_train = pd.read_csv("Data/train.csv")
    df_test = pd.read_csv("Data/test.csv")
    
    #df_encoded_train = process_features(df_train, df_test)
    #univariate_selection(df_encoded_train)
    #scatter_plot(df_train, 'SalePrice', 'PoolArea')
    box_plot(df_train, 'SalePrice')
    #dist_plot(df_train, 'OverallQual')
    #df_train, df_test = prune_features(df_train, df_test)