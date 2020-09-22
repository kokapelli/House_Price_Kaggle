import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.model_selection import cross_val_score, train_test_split



# root meansquare error - Cross validation
def rmse_cv(model, train, target):
    rmse = np.sqrt(-cross_val_score(model, train, target, scoring="neg_mean_squared_error", cv = 5))

    return(rmse)
