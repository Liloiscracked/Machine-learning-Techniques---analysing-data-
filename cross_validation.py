import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression

def nested_cross_validation(X, y, outer_folds=10, inner_folds=10):
    outer_cv = KFold(n_splits=outer_folds, shuffle=True, random_state=42)
    mae_scores = []
    rmse_scores = []

    for train_index, test_index in outer_cv.split(X):
        X_train_outer, X_test_outer = X.iloc[train_index], X.iloc[test_index]
        y_train_outer, y_test_outer = y.iloc[train_index], y.iloc[test_index]

        inner_cv = KFold(n_splits=inner_folds, shuffle=True, random_state=42)
        inner_mae_scores = []
        inner_rmse_scores = []

        for inner_train_index, inner_test_index in inner_cv.split(X_train_outer):
            X_train_inner, X_val_inner = X_train_outer.iloc[inner_train_index], X_train_outer.iloc[inner_test_index]
            y_train_inner, y_val_inner = y_train_outer.iloc[inner_train_index], y_train_outer.iloc[inner_test_index]

            model = LinearRegression()
            model.fit(X_train_inner, y_train_inner)

            y_pred_val = model.predict(X_val_inner)

            inner_mae = mean_absolute_error(y_val_inner, y_pred_val)
            inner_rmse = np.sqrt(mean_squared_error(y_val_inner, y_pred_val))

            inner_mae_scores.append(inner_mae)
            inner_rmse_scores.append(inner_rmse)

        avg_inner_mae = np.mean(inner_mae_scores)
        avg_inner_rmse = np.mean(inner_rmse_scores)

        model.fit(X_train_outer, y_train_outer)
        y_pred_outer = model.predict(X_test_outer)

        outer_mae = mean_absolute_error(y_test_outer, y_pred_outer)
        outer_rmse = np.sqrt(mean_squared_error(y_test_outer, y_pred_outer))

        mae_scores.append(outer_mae)
        rmse_scores.append(outer_rmse)

        print("Outer Fold MAE:", outer_mae)
        print("Outer Fold RMSE:", outer_rmse)

    avg_mae = np.mean(mae_scores)
    avg_rmse = np.mean(rmse_scores)

    print("Average MAE:", avg_mae)
    print("Average RMSE:", avg_rmse)

# Load data from CSV
data = pd.read_csv('Life Expectancy Data.csv')

# Split data into X and y
data = data.dropna()
X = data.drop(columns=['Life expectancy','Country','Year','Status'])
y = data['Life expectancy']

# Perform nested cross-validation
nested_cross_validation(X, y)
