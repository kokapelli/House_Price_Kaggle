# Scratchpad
[Excellent preprocessing descriptions](https://www.datacamp.com/community/tutorials/categorical-data)

## Potential step-by-step
1. Identify and Handle Null Value.
2. Identify and Handle Outliers.
3. Identify and Handle duplicates.
4. Feature Engineering.
5. Feature Importance.
6. Hyper tuning.
7. Cross-Validation.
8. Ensemble Learning.

## Important Factors
1. Location where the house is located has a huge impact on the pricing
2. Area, the size of the house has an immediate effect on the pricing
3. Building year, Depending on the location there can be a correlation. An old classic neighbourhood may have a higher price.

## **'SalePrice' : Target**

## Third feature culling
### Ongoing...
* *'MSSubClass'* : Identifies the type of dwelling 
* **'LotArea'**: Lot size in square feet
* **'Neighborhood'**: Physical locations within Ames city limits
* **'BldgType'**: Type of dwelling
* **'HouseStyle'**: Style of dwelling
* **'OverallQual'**: Rates the overall material and finish of the house
* **'OverallCond'**: Rates the overall condition of the house
* **'YearBuilt'**: Original construction date
* **'GarageType'**: Garage location
* **'GarageCars'**: Size of garage in car capacity
* **'GarageArea'**: Size of garage in square feet
* **'PoolArea'**: Pool area in square feet
* **'SaleCondition'**: Condition of Sale


## Second feature culling
* *'MSSubClass'* : Identifies the type of dwelling 
* **'LotArea'**: Lot size in square feet
* **'Neighborhood'**: Physical locations within Ames city limits
* **'BldgType'**: Type of dwelling
* **'HouseStyle'**: Style of dwelling
* **'OverallQual'**: Rates the overall material and finish of the house
* **'OverallCond'**: Rates the overall condition of the house
* **'YearBuilt'**: Original construction date
* **'GarageType'**: Garage location
* **'GarageCars'**: Size of garage in car capacity
* **'GarageArea'**: Size of garage in square feet
* **'PoolArea'**: Pool area in square feet
* **'SaleCondition'**: Condition of Sale


## First Feature Culling
* **'MSSubClass'** : Identifies the type of dwelling involved in the sale.
* 'MSZoning' : Identifies the general zoning classification of the sale.
..* Depends on their definition of for example industrial, agricultural etc. Is industry an entire factory? Is agricultural the smallest barn?
* 'LotFrontage': Linear feet of street connected to property
* **'LotArea'**: Lot size in square feet
* *'Street'*: Type of road access to property
* *'Alley'*: Type of alley access to property
* *'LotShape'*: General shape of property
* *'LandContour'*: Flatness of the property
* 'Utilities': Type of utilities available
..* Depends on the data and how common it is that it lacks utilities
* *'LotConfig'*: Lot configuration
* *'LandSlope'*: Slope of property
* **'Neighborhood'**: Physical locations within Ames city limits
* *'Condition1'*: Proximity to various conditions
* *'Condition2'*: Proximity to various conditions (If more than one is present)
* **'BldgType'**: Type of dwelling
* **'HouseStyle'**: Style of dwelling
* **'OverallQual'**: Rates the overall material and finish of the house
* **'OverallCond'**: Rates the overall condition of the house
* **'YearBuilt'**: Original construction date
* 'YearRemodAdd': Remodel date (same as construction date if no remodeling or additions)
* *'RoofStyle'*: Type of roof
* *'RoofMatl'*: Roof material
* *'Exterior1st'*: Exterior covering on house
* *'Exterior2nd'*: Exterior covering on house (If more than one)
* *'MasVnrType'*
* *'MasVnrArea'*
* 'ExterQual': Evaluates the quality of the material on the exterior
* 'ExterCond': Evaluates the present condition of the material on the exterior
* *'Foundation'*
* *'BsmtQual'*
* *'BsmtCond'*
* *'BsmtExposure'*
* *'BsmtFinType1'*
* *'BsmtFinSF1'*
* *'BsmtFinType2'*
* *'BsmtFinSF2'*
* *'BsmtUnfSF'*
* *'TotalBsmtSF'*
* *'Heating'*
* *'HeatingQC'*
* *'CentralAir'*
* *'Electrical'*
* *'1stFlrSF'*
* *'2ndFlrSF'*
* *'LowQualFinSF'*
* *'GrLivArea'*
* *'BsmtFullBath'*
* *'BsmtHalfBath'*
* *'FullBath'*
* *'HalfBath'*
* *'BedroomAbvGr'*
* *'KitchenAbvGr'*
* *'KitchenQual'*
* *'TotRmsAbvGrd'*
* *'Functional'*
* *'Fireplaces'*
* *'FireplaceQu'*
* **'GarageType'**: Garage location
* *'GarageYrBlt'*
* *'GarageFinish'*
* **'GarageCars'**: Size of garage in car capacity
* **'GarageArea'**: Size of garage in square feet
* *'GarageQual'*
* *'GarageCond'*
* *'PavedDrive'*
* *'WoodDeckSF'*
* *'OpenPorchSF'*
* *'EnclosedPorch'*
* *'3SsnPorch'*
* *'ScreenPorch'*
* **'PoolArea'**: Pool area in square feet
* *'PoolQC'*
* *'Fence'*
* 'MiscFeature': Miscellaneous feature not covered in other categories
* *'MiscVal'*
* *'MoSold'*
* *'YrSold'*
* 'SaleType': Type of sale
* **'SaleCondition'**: Condition of Sale

[MD Cheat sheet](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet)