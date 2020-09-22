import pandas as pd

from categorical_mappings import *
from feature_bins import *
from plotting import *
from preprocessing import *
from model import*

from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb


TRAIN = False
TRAIN2 = True

def process_features(train, test):
    cat_columns = convert_object_to_category(train)
    labels      = get_classes(cat_columns)
    label_dicts = cateogrical_to_numerical(train, labels)
    enc_train   = map_categorical_to_numerical(train, label_dicts)
    
    return enc_train

def refactor_features(df):
    df = df.drop(['Utilities', 'Street'], axis=1)
    
    # General None and Zero fills
    na_feats = ["MasVnrType", 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', "MiscFeature", "Fence", "FireplaceQu", "PoolQC", "Alley", 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
    df = fill_na_to_none(df, na_feats)
    zero_feats = ["MasVnrArea", 'GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']
    df = fill_na_to_zero(df, zero_feats)

    # Specific fills
    df['MSZoning']    = df['MSZoning'].fillna(df['MSZoning'].mode()[0])
    df["Functional"]  = df["Functional"].fillna("Typ")
    df['Electrical']  = df['Electrical'].fillna(df['Electrical'].mode()[0])
    df['KitchenQual'] = df['KitchenQual'].fillna(df['KitchenQual'].mode()[0])
    df['Exterior1st'] = df['Exterior1st'].fillna(df['Exterior1st'].mode()[0])
    df['Exterior2nd'] = df['Exterior2nd'].fillna(df['Exterior2nd'].mode()[0])
    df['SaleType']    = df['SaleType'].fillna(df['SaleType'].mode()[0])

    # Merging and Feature Creation
    df['YearsSinceRemodel']     = df['YrSold'].astype(int) - df['YearRemodAdd'].astype(int)
    df['Total_Home_Quality']    = df['OverallQual'] + df['OverallCond']
    df['TotalSF']               = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']

    df['Total_sqr_footage'] = (df['BsmtFinSF1'] + df['BsmtFinSF2'] + df['1stFlrSF'] + df['2ndFlrSF'])
    df['MasVnrArea']        = df['MasVnrArea'].apply(lambda x: np.exp(4) if x <= 0.0 else x)
    df['BsmtFinSF1']        = df['BsmtFinSF1'].apply(lambda x: np.exp(6.5) if x <= 0.0 else x)
    df['Total_Bathrooms']   = (df['FullBath'] + (0.5 * df['HalfBath']) + df['BsmtFullBath'] + (0.5 * df['BsmtHalfBath']))
    df['TotalBsmtSF']       = df['TotalBsmtSF'].apply(lambda x: np.exp(6) if x <= 0.0 else x)
    df["LotFrontage"]       = df.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

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

    # Testing out outlier function
    num_col  = df_train.loc[:,'MSSubClass':'SaleCondition'].select_dtypes(exclude=['object']).columns
    df_train = detect_outliers(df_train, 2, num_col, drop=True)

    # Quick check over missing values and features
    #print(df_train.info())

    # in case of positive skewness, log transformations usually works well.
    df_train['SalePrice'] = np.log(df_train['SalePrice'])

    all_features = combine_train_test(df_train, df_test, 'SalePrice')

    all_features = refactor_features(all_features)
    #all_features['GrLivArea'] = np.log(all_features['GrLivArea'])
    #all_features.loc[all_features['HasBsmt']==1,'TotalBsmtSF'] = np.log(all_features['TotalBsmtSF'])

    # Process chosen features
    all_features = log_skewed_feats(all_features, 0.75)
    # Convert categorical variable into dummy
    all_features = pd.get_dummies(all_features)
    all_features = all_features.fillna(all_features.mean())

    #creating matrices for sklearn:
    X_train = all_features[:df_train.shape[0]]
    X_test = all_features[df_train.shape[0]:]
    y = df_train.SalePrice

    # Modelling
    model_ridge = Ridge()
    model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, y)

    alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
    cv_ridge = [rmse_cv(Ridge(alpha = alpha), X_train, y).mean() 
                for alpha in alphas]

    #cv_ridge = pd.Series(cv_ridge, index = alphas)
    #cv_ridge.plot()
    #plt.xlabel("alpha")
    #plt.ylabel("rmse")
    #plt.show()

    model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, y)
    print(rmse_cv(model_lasso, X_train, y).mean())

    if(TRAIN):
        dtrain = xgb.DMatrix(X_train, label = y)
        dtest = xgb.DMatrix(X_test)

        params = {"max_depth":2, "eta":0.1}
        model = xgb.cv(params, dtrain,  num_boost_round=500, early_stopping_rounds=100)

        model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1) #the params were tuned using xgb.cv
        model_xgb.fit(X_train, y)
        xgb_preds = np.expm1(model_xgb.predict(X_test))
        lasso_preds = np.expm1(model_lasso.predict(X_test))
        predictions = pd.DataFrame({"xgb":xgb_preds, "lasso":lasso_preds})

        preds = 0.7*lasso_preds + 0.3*xgb_preds
        solution = pd.DataFrame({"id":df_test.Id, "SalePrice":preds})
        solution.to_csv("ridge_sol.csv", index = False)

    if(TRAIN2):
        params = {
        'objective':'reg:linear',
        'booster':'gbtree',
        'max_depth':2,
        'eval_metric':'rmse',
        'learning_rate':0.07, 
        'min_child_weight':1,
        'subsample':0.80,
        'colsample_bytree':0.81,
        'seed':45,
        'reg_alpha':1,
        'reg_lambda':1,
        'gamma':0,
        'nthread':-1

    }

    x_train, x_valid, y_train, y_valid = train_test_split(X_train, y, test_size=0.2, random_state=10)

    d_train = xgb.DMatrix(x_train, label=y_train)
    d_valid = xgb.DMatrix(x_valid, label=y_valid)
    d_test = xgb.DMatrix(X_test)

    watchlist = [(d_train, 'train'), (d_valid, 'valid')]

    clf = xgb.train(params, d_train, 2000, watchlist, early_stopping_rounds=400, maximize=False, verbose_eval=10)

    p_test = clf.predict(d_test)
    d_test = clf.predict(d_valid)
    expo_p_test = np.expm1(p_test)
    
    solution = pd.DataFrame({"id":df_test.Id, "SalePrice":expo_p_test})
    solution.to_csv("ridge_sol2.csv", index = False)