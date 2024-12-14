import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import sklearn.metrics as mt
import matplotlib.pyplot as plt

file_path = r"D:\qusay-alkaleh\study\maki̇ne-ogrenmesi\project\dataset_preprocessed.csv"
df = pd.read_csv(file_path)

X = df.drop('Price' , axis=1)
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.30, 
                                                    random_state=42)

model = SVR(kernel='rbf', C=0.1, gamma=0.1)

model.fit(X_train, y_train)

# Train performance
print("### Training Performance for SVR ###")
predicted_y_train = model.predict(X_train)

# Assuming y_true and y_pred are actual and predicted values
mae = mt.mean_absolute_error(y_train, predicted_y_train)
mse = mt.mean_squared_error(y_train, predicted_y_train)
rmse = mt.root_mean_squared_error(y_train, predicted_y_train)
r2 = mt.r2_score(y_train, predicted_y_train)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R-squared: {r2}")
print("\n")

# Test performance
print("### Testing Performance for SVR ###")
predicted_y_test = model.predict(X_test)

# Assuming y_true and y_pred are actual and predicted values
mae = mt.mean_absolute_error(y_test, predicted_y_test)
mse = mt.mean_squared_error(y_test, predicted_y_test)
rmse = mt.root_mean_squared_error(y_test, predicted_y_test)
r2 = mt.r2_score(y_test, predicted_y_test)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R-squared: {r2}")

# Visualization
plt.figure(figsize=(10, 5))
plt.scatter(y_test, predicted_y_test, color='blue', alpha=0.6, label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal Fit')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title("SVR - Predicted vs Actual")
plt.legend()
plt.show()