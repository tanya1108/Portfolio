import pandas as pd
from jovian.testing import testcase

dataset_cols = ['Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
       'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
       'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
       'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
       'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',
       'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',
       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
       'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
       'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',
       'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',
       'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',
       'SaleCondition', 'SalePrice']

categorical_cols = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 
        'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 
        'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 
        'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 
        'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 
        'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 
        'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']


def are_preds_good(nb):
    try:
        train_rmse = nb.value("mean_squared_error(train_targets, train_preds, squared=False)")
        train_allclose = nb.value("np.allclose(train_targets, train_preds, 10)")
        val_allclose = nb.value("np.allclose(val_targets, val_preds, 10)")
        val_rmse = nb.value("mean_squared_error(val_targets, val_preds, squared=False)")
        return train_rmse < 50000 and val_rmse < 50000 and not train_allclose and not val_allclose
    except Exception:
        return False


@testcase("Q1", "train.csv wasn't loaded properly")
def test_q1(nb):
    if not are_preds_good(nb):
        assert list(nb.value("prices_df.shape")) == [1460, 81], (
            "'prices_df' doesn't have the expected number of rows & columns")
        
        assert nb.value("prices_df.columns.tolist()") == dataset_cols, (
            "'prices_df' doesn't have the expected column names")


@testcase("Q2", "The value of 'n_rows' or 'n_cols' is incorrect")
def test_q2(nb):
    if not are_preds_good(nb):
        assert nb['n_rows'] == 1460, (
            "'n_rows' is incorrect")
        
        assert nb['n_cols'] == 81, (
            "'n_cols' is incorrect")


@testcase("Q3", "The values of 'input_cols' or 'target_col' is incorrect")
def test_q3(nb):
    assert not 'SalePrice' in nb.get('list(input_cols)'), "Target column 'SalePrice' should not be a part of 'input_cols'"

    if not are_preds_good(nb):
        assert nb.get("list(input_cols)").sort() == dataset_cols[1:-1].sort(), (
            "'input_cols' must be a list of strings. It should contain names of input columns")

        assert nb.get("target_col") == "SalePrice" or nb.get("target_col") == ["SalePrice"], (
            "'target_col' must be a string matching the name of the column we're trying to predict")


@testcase("Q4", "The value of categorical_cols is incorrect")
def test_q4(nb):
    if not are_preds_good(nb):
        assert nb['list(categorical_cols)'].sort() == categorical_cols.sort(), (
            "'categorical_cols' must be a list of strings, containing names of categorical columns")

        assert nb['len(categorical_cols)'] == 43, (
            "'categorical_cols' must contain 43 column names")


@testcase("Q5", "Imputation of missing values in numeric columns was not performed properly")
def test_q5(nb):
    if not are_preds_good(nb):
        assert nb.get('isinstance(imputer, SimpleImputer)'), (
            "'imputer' must be an instance of 'SimpleImputer'")
        
        assert nb.get('len(imputer.statistics_)') == 36, (
            "The imputer was not fitted properly to the numeric columns. "
            "'imputer.statistics_' must contain 36 values.")
        
        check = 'len(inputs_df[numeric_cols].isna().sum().sort_values(ascending=False).loc[lambda x: x > 0])'
        assert nb.get(check) == 0, (
            "The numeric columns were not imputed properly. "
            "Numeric columns must not contain any missing values after imputation")


@testcase("Q6", "Scaling of values in numeric columns was not peformed properly")
def test_q6(nb):
    if not are_preds_good(nb):
        assert nb.get("isinstance(scaler, MinMaxScaler)"), (
            "'scaler' must be an instance of 'MinMaxScaler'")
        
        assert nb.get('len(scaler.data_max_)') == 36, (
            "The scaler was not fitted properly to the numeric columns. "
            "'scaler.data_max_' must contain 36 values.")

        is_scaled = nb.get("all(inputs_df[numeric_cols].describe().loc['max'] < 2)") and \
            nb.get("all(inputs_df[numeric_cols].describe().loc['min'] >= -1)")
        
        assert is_scaled, (
            "The numeric columns were not scaled properly. " 
            "Values in numeric columns must all lie in the range (0,1)")


@testcase("Q7", "One hot encoding of categorical columns was not performed properly")
def test_q7(nb):
    if not are_preds_good(nb):
        assert nb.get("isinstance(encoder, OneHotEncoder)"), (
            "'encoder' must be an instance of 'OneHotEncoder'")
        
        assert nb.get('len(encoder.categories_)') == 43, (
            "The encoder was not fitted properly to the categorical columns. "
            "'encoder.categories_' must contain 43 elements (one for each categorical column) ")
        
        assert nb.get('set(encoded_cols).issubset(inputs_df.columns)'), (
            "'encoded_cols' must be a a subset of 'inputs_df.columns' after encoding")
        
        assert nb.get('int(inputs_df[encoded_cols].nunique().mean())') <= 2, (
            "Each column from 'encoded_cols' must contain only 1s and 0s, no other values")


@testcase("Q8", "The ridge linear regression model was not trained properly")
def test_q8(nb):
    if not are_preds_good(nb):
        assert nb.get("isinstance(model, Ridge)"), (
            "'model' must be an instance of 'Ridge'")

        assert nb.get("len(model.coef_.flatten().tolist())") == 304, (
            "After fitting, 'model.coef_' must be a list of 304 weights")


@testcase("Q9", "Predictions on the training and validation aren't done correctly")
def test_q9(nb):
    if not are_preds_good(nb):
        assert nb.get("len(train_preds)") == 1095, (
            "'train_preds' must be a vector containing 1095 numbers")
        
        assert nb.get("len(val_preds)") == 365, (
            "'val_preds' must be a vector containing 365 numbers")
        
        assert nb.get("train_rmse") < 50000, (
            "'train_rmse' must be under 50000 otherwise your model isn't very useful "
            "(make sure to pass the argument squared=False to 'mean_squared_error')")
        
        assert nb.get("val_rmse") < 50000, (
            "'val_rmse' must be under 50000 otherwise your model isn't very useful "
            "(make sure to pass the argument squared=False to 'mean_squared_error')")


@testcase("Q10", "The value of 'weights' was incorrect")
def test_q10(nb):
    if not are_preds_good(nb):
        assert nb.get('len(weights)') == 304, (
            "'weights' must be a list 304 numbers that are used as "
            "coefficients by the linear regression model.")