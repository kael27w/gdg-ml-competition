import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error
from feature_scaling import prepare_features

# Loading our dataset
# Change this line in XGBoost.py
df = pd.read_csv("data/IMDB top 1000.csv")  # Update the path to your actual CSV file


# Getting the Preprocessed and scaled data.
X, y = prepare_features(df)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)


# Define a custom callback class to track the training R-squared score
train_r2_scores = []


class TrackR2Score(xgb.callback.TrainingCallback):
    def after_iteration(self, model, epoch, evals_log):
        # Calculate the training R-squared score
        pred = model.predict(xgb.DMatrix(X_train, label=y_train))
        train_r2 = r2_score(y_train, pred)
        train_r2_scores.append(train_r2)


param_grid = {
    "n_estimators": [100, 500],
    "max_depth": [3, 6],
    "learning_rate": [0.05, 0.1],
}

grid_search = GridSearchCV(
    estimator=xgb.XGBRegressor(
        objective="reg:squarederror", random_state=42, callbacks=[TrackR2Score()]
    ),
    param_grid=param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1,
)
grid_search.fit(X, y)
best_params = grid_search.best_params_
best_score = grid_search.best_score_
print("Best Parameters:", best_params)
print("Best R^2 Score:", best_score)

best_model = xgb.XGBRegressor(
    objective="reg:squarederror", random_state=42, **best_params
)
best_model.fit(X_train, y_train)
train_predictions = best_model.predict(X_train)
test_predictions = best_model.predict(X_test)


def calculate_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    msle = mean_squared_log_error(y_true, y_pred)
    mape = np.mean(np.abs((np.exp(y_true) - np.exp(y_pred)) / np.exp(y_true))) * 100
    return r2, mse, msle, mape


train_r2, train_mse, train_msle, train_mape = calculate_metrics(
    y_train, train_predictions
)
test_r2, test_mse, test_msle, test_mape = calculate_metrics(y_test, test_predictions)

print(f"\nTraining Metrics:")
print(f"R2 score: {train_r2:.4f}")
print(f"MSE: {train_mse:.4f}")
print(f"MLSE: {train_msle:.4f}")
print(f"MAPE: {train_mape:.2f}%")

print(f"\nTest Metrics:")
print(f"R2 score: {test_r2:.4f}")
print(f"MSE: {test_mse:.4f}")
print(f"MSLE: {test_msle:.4f}")
print(f"MAPE: {test_mape:.2f}%")

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_train, train_predictions, color="blue", label="Train")
plt.scatter(y_test, test_predictions, color="red", label="Test")
plt.title("Actual vs Predicted Values")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.legend()
plt.show()

# Plot the training R-squared score curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_r2_scores) + 1), train_r2_scores)
plt.title("Training R-squared Score Curve")
plt.xlabel("Iterations")
plt.ylabel("R-squared Score")
plt.show()


# Feature importance
feature_importance = best_model.feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + 0.5

plt.figure(figsize=(12, 8))
plt.barh(pos, feature_importance[sorted_idx], align="center")
plt.yticks(pos, X.columns[sorted_idx])
plt.xlabel("Feature Importance")
plt.title("Variable Importance")
plt.tight_layout()
plt.show()