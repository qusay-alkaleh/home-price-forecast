import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

file_path = r"D:\qusay-alkaleh\study\maki̇ne-ogrenmesi\project\dataset_scaled.csv"
df = pd.read_csv(file_path)

# Assuming df is your dataframe and the required columns are already defined
X = df.drop('Price', axis=1)
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=0.75, 
                                                    random_state=42)

# KNN model
knn_model = KNeighborsRegressor(n_neighbors=5)  # You can adjust n_neighbors as needed
knn_model.fit(X_train, y_train)

# Predictions for KNN model
knn_predictions = knn_model.predict(X_test)

# Evaluate KNN model
print("KNN Performance:")
print(f"Training score of KNN: {knn_model.score(X_train, y_train)}")
print(f"R^2 Score: {r2_score(y_test, knn_predictions)}")
print(f"Mean Squared Error: {mean_squared_error(y_test, knn_predictions)}")

# Plot predicted values vs actual target values
plt.figure(figsize=(10,6))
plt.scatter(y_test, knn_predictions, color='blue', alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)
plt.title('KNN: Predicted vs Actual Prices')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.grid(True)
plt.show()