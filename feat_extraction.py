import pandas as pd
import numpy as np

from plotting import *
from categorical_mappings import *
from feature_bins import *


def process_cols(df):
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
    train = process_cols(train)
    test = process_cols(test)

    return train, test

if __name__ == "__main__":
    df_train = pd.read_csv("Data/train.csv")
    df_test = pd.read_csv("Data/test.csv")
    
    df_train, df_test = prune_features(df_train, df_test)