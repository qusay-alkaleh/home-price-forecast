import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

file_path = r"D:\qusay-alkaleh\study\maki̇ne-ogrenmesi\project\dataset_labeled.csv"
df = pd.read_csv(file_path)

X = df.drop('Price' , axis=1)
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                    train_size = 0.75, 
                                                    random_state = 42)

# Random Forest model
random_forest_model = RandomForestRegressor()
random_forest_model.fit(X_train, y_train)

# Predictions for Random Forest model
random_forest_predictions = random_forest_model.predict(X_test)

# Evaluate Random Forest model
print("Random Forest Performance:")
print(f"Training score of random forest: {random_forest_model.score(X_train, y_train)}")
print(f"R^2 Score: {r2_score(y_test, random_forest_predictions)}")
print(f"Mean Squared Error: {mean_squared_error(y_test, random_forest_predictions)}")

# Plot predicted values vs actual target values
plt.figure(figsize=(10,6))
plt.scatter(y_test, random_forest_predictions, color='blue', alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)
plt.title('Random Forest: Predicted vs Actual Prices')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.grid(True)
plt.show()