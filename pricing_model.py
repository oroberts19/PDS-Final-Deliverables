
# importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor

import category_encoders
from category_encoders import TargetEncoder, LeaveOneOutEncoder

import shap
import seaborn as sns

# reading in the data
df = pd.read_csv('sample_pricing.csv', index_col=0) # your working directory here

# checking how many NA we have
df.isna().sum()
len(df)

# replacing NAs, empty strings, or other placeholder inputs with 'unknown' so that it is its own category
df['trim'] = df['trim'].str.replace('[#,@,&,!,~]','')
df['trim'] = df['trim'].replace([" "], 'unknown')
df['trim'] = df['trim'].replace(np.nan, 'unknown')
df['interior_color'] = df['interior_color'].replace(np.nan, 'unknown')
df['engine'] = df['engine'].replace(np.nan, 'unknown')
df['interior_color'] = df['interior_color'].replace(["'", "(1)", "**see photos**", "*see photos*",
                                                     ",", "--", "-1", ",graph w"], 'unknown')

# checking NAs again 
df.isnull().sum()
df1 = df

# imputing NAs in the column year by filling in the median year 
df1['year'].fillna((df1['year'].median()), inplace = True)

# the remaining NAs that were not addressed are dropped
df1 = df1.dropna()

# saving our numerical features to a newc variable
subset = df1[['year', 'used', 'certified', 'mileage', 'price', 'median_income', 'number_days_posted', 'sold']]

# pairplot of numerical features (will take long if run on entire dataset)
"""
sns.pairplot(subset)
"""

# heatmap of numerical features (will take long if run on entire dataset)
"""
corr = subset.corr()
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, cmap=sns.diverging_palette(220, 10, as_cmap=True))
"""

# checking skewness of our features
df1.skew()

"""# Preprocessing"""

# defining X and Y for the model
Y = df1['price']
X =  df1[['year', 'used', 'certified', 'mileage',
          'make','model', 'trim', 'body', 'exterior_color', 'interior_color',
          'engine', 'name', 'city', 'state', 'zip_code',
          'median_income', 'number_days_posted', 'sold']]

# train and test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

# target encoding our categorical variables
enc = TargetEncoder(cols=['make','model', 'trim', 'body', 'exterior_color', 'interior_color', 'engine', 'name', 'city', 'state', 'zip_code'])
transformed_df_train = enc.fit_transform(X_train,Y_train)
transformed_df_test = enc.fit_transform(X_test,Y_test)

"""# Modelling """

# defining again the train and test X after the encoding transformation
X_train1 =  transformed_df_train[['year', 'used', 'certified', 'mileage',
          'make','model', 'trim', 'body', 'exterior_color', 'interior_color',
          'engine', 'name', 'city', 'state', 'zip_code',
          'median_income', 'number_days_posted', 'sold']]

X_test1 = transformed_df_test[['year', 'used', 'certified', 'mileage',
          'make','model', 'trim', 'body', 'exterior_color', 'interior_color',
          'engine', 'name', 'city', 'state', 'zip_code',
          'median_income', 'number_days_posted', 'sold']]

"""### Random Forest"""

# defining the RF model and fitting it (will take long if run on entire dataset)
model = RandomForestRegressor(random_state=0)
model.fit(X_train1, Y_train)

# using it on the test to make predictions
y_pred_rf = model.predict(X_test1)

# defining a function to calculate the MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# getting the R squared for the RF
print('R-squared:', r2_score(Y_test, y_pred_rf))

print('Random forest MAE:', metrics.mean_absolute_error(Y_test, y_pred_rf))
print('Random forest MSE:', metrics.mean_squared_error(Y_test, y_pred_rf))
print('Random forest RMSE:', np.sqrt(metrics.mean_squared_error(Y_test, y_pred_rf)))

# getting the MAPE for the RF
mean_absolute_percentage_error(Y_test, y_pred_rf)

# using SHAP to get the feature importance plot (will take long if run on entire dataset)
"""
shap_values = shap.TreeExplainer(model).shap_values(X_test1)
shap.summary_plot(shap_values, X_test1, plot_type="bar")
"""

"""### Multiple Linear Regression"""

# starting with baseline model: Dummy Regression, it predicts the mean every time
dummy_mean = DummyRegressor(strategy='mean')
dummy_mean.fit(X_train1, Y_train)
DummyRegressor(constant=None, quantile=None, strategy='mean')

# defining our MLR model and fitting it
lin_reg_mod = LinearRegression()
lin_reg_mod.fit(X_train1, Y_train)

# getting the coefficients and intercept of the regression model
print(lin_reg_mod.intercept_)
print(lin_reg_mod.coef_)

# r squared of dummy model
dummy_mean.score(X_test1, Y_test)

# r squared of MLR
lin_reg_mod.score(X_test1, Y_test)

# making predictions on test set
y_pred = lin_reg_mod.predict(X_test1)
y_pred_dummy_mean = dummy_mean.predict(X_test1)

# errors of dummy model
print('Dummy MAE:', metrics.mean_absolute_error(Y_test, y_pred_dummy_mean))
print('Dummy RMSE:', np.sqrt(metrics.mean_squared_error(Y_test, y_pred_dummy_mean)))
print('Dummy MAPE:', mean_absolute_percentage_error(Y_test, y_pred_dummy_mean))

# errors of MLR 
print('Linear regression MAE:', metrics.mean_absolute_error(Y_test, y_pred))
print('Linear regression RMSE:', np.sqrt(metrics.mean_squared_error(Y_test, y_pred)))
print('Linear regression MAPE:', mean_absolute_percentage_error(Y_test, y_pred))