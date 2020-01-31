import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from feat_extraction import *

# Advanced regression techniques like random forest and gradient boosting

if __name__ == "__main__":
    train = pd.read_csv("/Users/Kukus/Desktop/House_Prices_Kaggle/Data/train.csv")
    test = pd.read_csv("/Users/Kukus/Desktop/House_Prices_Kaggle/Data/test.csv")
    train, test = prune_features(train, test)