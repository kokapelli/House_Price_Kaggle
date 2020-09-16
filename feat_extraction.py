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
    
    df = df.drop(z_score(df, 'LotArea', 3))

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
    ALL_FEATURES = ['Exterior1st', 'Exterior2nd', 'MasVnrType',  'ExterQual',
       'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
       'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical',
       '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',  'BsmtFullBath',
       'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
       'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
       'ScreenPorch',   'MiscFeature',  
        'SaleType', 'SaleCondition',  'BsmtFinType1_Unf']

    CHOSEN_FEATURES = ['GrLivArea','TotalBsmtSF','TotalSF','Total_Home_Quality', 'Total_sqr_footage']
    OBSOLETE_FEATURES = ['MiscVal','MSSubClass', 'MSZoning',   'Alley','Condition1', 'Condition2', 'BldgType', 'HouseStyle',  'YearRemodAdd', 'RoofStyle', 'RoofMatl','LotShape', 'LandContour', 'LotConfig', 'LandSlope','LotFrontage','LotArea','Functional', 'Fireplaces','PavedDrive','GarageCars',  'GarageQual', 'GarageCond','GarageType',  'GarageFinish','GarageYrBlt','GarageArea','FireplaceQu','KitchenQual','Neighborhood','YearBuilt','MasVnrArea','TotRmsAbvGrd', 'PoolArea','OverallQual','OverallCond''HasWoodDeck', 'HasEnclosedPorch', 'Has3SsnPorch','HasScreenPorch', 'Total_porch_sf', 'HasFireplace','HasBsmt','HasGarage','Fence','MoSold','HasPool','YrSold','SalePrice','HasOpenPorch','YrBltAndRemod', 'YearsSinceRemodel','Id', 'Total_Bathrooms',  'Has2ndFloor']

    df_train = pd.read_csv("Data/train.csv")
    df_test = pd.read_csv("Data/test.csv")
    var = 'SalePrice'

    # Remove the IDs
    train_ID = df_train['Id']
    test_ID = df_test['Id']
    df_train.drop(['Id'], axis=1, inplace=True)
    df_test.drop(['Id'], axis=1, inplace=True)

    #print(df_train.iloc[:, 0:10])
    
    #df_train = refactor_features(df_train)
    #print(df_train[var].value_counts())
    plot_every_numeric(df_train, 'SalePrice')
    #plot_all_corr_heatmap(df_train, 'SalePrice')

    #df_train = outlier_processing(df_train)
    #df_train = df_train.drop(z_score(df_train, 'LotArea', 3))

    #scatter_plot(df_train, 'SalePrice', var)
    #box_plot(df_train, 'SalePrice', var)
    #dist_plot(df_train, var, True)


    #https://www.kaggle.com/lavanyashukla01/how-i-made-top-0-3-on-a-kaggle-competition