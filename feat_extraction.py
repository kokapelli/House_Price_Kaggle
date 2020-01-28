import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


def drop_columns(df, columns):
    df = df.drop(columns, axis=1)
    return df


def get_classes(df):
    return list(df.columns.values)

if __name__ == "__main__":
    train = pd.read_csv("/Users/Kukus/Desktop/House_Prices_Kaggle/Data/train.csv")
    test = pd.read_csv("/Users/Kukus/Desktop/House_Prices_Kaggle/Data/test.csv")
    
    """
    columns_to_drop = ['Example']
    drop_columns(train, columns_to_drop)
    drop_columns(test, columns_to_drop)
    """

    print(train.head())
    print(get_classes(train))