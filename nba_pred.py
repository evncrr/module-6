import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

#load data
df = pd.read_csv("NBA/nba_2022-23_all_stats_with_salary.csv")
df = df.dropna()

#features and target
X = df[['PTS', 'AST', 'TOV', 'TRB', 'STL', 'BLK', '2P%', '3P%', 'FT%', 'PER', 'WS', 'BPM', 'Age']]
y = df['Salary']

#split data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#initialize Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

#train model
rf_model.fit(X_train, y_train)

#make predictions on the training and test data
y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)

#evaluate model
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

#results
print(f'Train MAE: ${train_mae:,.2f}')
print(f'Test MAE: ${test_mae:,.2f}')
print(f'Train R²: {train_r2:.2f}')
print(f'Test R²: {test_r2:.2f}')

#find top 5 worst predictions
errors = np.abs(y_test_pred - y_test.to_numpy())
top_5_errors = errors.argsort()[-5:]  #indices of worst predictions

for idx in top_5_errors:
    print(f"Predicted: {y_test_pred[idx]:,.0f} | Actual: {y_test.iloc[idx]:,.0f}")

#plot predicted vs actual
plt.scatter(y_test, y_test_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel('Actual Salary')
plt.ylabel('Predicted Salary')
plt.title('Predicted vs Actual Salary')
plt.show()