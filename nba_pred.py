from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

#load data
df = pd.read_csv("NBA/nba_2022-23_all_stats_with_salary.csv")
df = df.dropna()

#features and target
X = df[['PTS', 'AST', 'TOV', 'TRB', 'STL', 'BLK', '2P%', '3P%', 'FT%', 'PER', 'WS', 'BPM', 'Age']]
y = df['Salary']

#normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

#KNN Regression
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)

#model evaluation
train_preds = knn.predict(X_train)
train_mae = mean_absolute_error(y_train, train_preds)
train_r2 = r2_score(y_train, train_preds)
test_preds = knn.predict(X_test)
test_mae = mean_absolute_error(y_test, test_preds)
test_r2 = r2_score(y_test, test_preds)

print(f"Train MAE: {train_mae:.2f}, Train R²: {train_r2:.2f}")
print(f"Test MAE: {test_mae:.2f}, Test R²: {test_r2:.2f}")

#find top 5 worst predictions
errors = np.abs(test_preds - y_test.to_numpy())
top_5_errors = errors.argsort()[-5:]  #indices of worst predictions
for idx in top_5_errors:
    print(f"Predicted: {test_preds[idx]:,.0f} | Actual: {y_test.iloc[idx]:,.0f}")

#plot predicted vs actual
plt.scatter(y_test, test_preds)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel('Actual Salary')
plt.ylabel('Predicted Salary')
plt.title('Predicted vs Actual Salary')
plt.show()