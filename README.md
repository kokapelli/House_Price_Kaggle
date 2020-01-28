## [Challenge](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview)

Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. But this playground competition's dataset proves that much more influences price negotiations than the number of bedrooms or a white-picket fence.

With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home.

# Goal
It is your job to predict the sales price for each house. For each Id in the test set, you must predict the value of the SalePrice variable. 

# Metric
Submissions are evaluated on Root-Mean-Squared-Error (RMSE) between the logarithm of the predicted value and the logarithm of the observed sales price. (Taking logs means that errors in predicting expensive houses and cheap houses will affect the result equally.)

# Important Factors
1. Location, where the house is located has a huge impact on the pricing
2. Area, the size of the house has an immediate effect on the pricing
3. Building year, Depending on the location there can be a correlation. An old classic neighbourhood may have a higher price.
