from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def nested_cross_validation(X, y, outer_folds=10, inner_folds=10):
    model = LinearRegression()

    outer_cv = KFold(n_splits=outer_folds, shuffle=True, random_state=42)

    mae_scores = []
    rmse_scores = []
    for fold_idx, (train_index, test_index) in enumerate(outer_cv.split(X)):

        X_train_outer, X_test_outer = X[train_index], X[test_index]
        y_train_outer, y_test_outer = y[train_index], y[test_index]

        inner_cv = KFold(n_splits=inner_folds, shuffle=True, random_state=42)

        # Initialize lists to store inner fold scores
        inner_mae_scores = []
        inner_rmse_scores = []

        for train_index_inner, val_index_inner in inner_cv.split(X_train_outer):
            X_train_inner, X_val_inner = X_train_outer[train_index_inner], X_train_outer[val_index_inner]
            y_train_inner, y_val_inner = y_train_outer[train_index_inner], y_train_outer[val_index_inner]

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

        print("Fold %d: MAE: %.4f, RMSE: %.4f" % (fold_idx, outer_mae, outer_rmse))

    avg_mae = np.mean(mae_scores)
    avg_rmse = np.mean(rmse_scores)

    return avg_mae, avg_rmse