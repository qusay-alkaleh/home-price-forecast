import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
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

# Gradient Boosting model
gradient_boosting_model = GradientBoostingRegressor()
gradient_boosting_model.fit(X_train, y_train)

# Predictions for Gradient Boosting model
gradient_boosting_predictions = gradient_boosting_model.predict(X_test)

# Evaluate Gradient Boosting model
print("Gradient Boosting Performance:")
print(f"Training score of gradient boosting: {gradient_boosting_model.score(X_train, y_train)}")
print(f"R^2 Score: {r2_score(y_test, gradient_boosting_predictions)}")
print(f"Mean Squared Error: {mean_squared_error(y_test, gradient_boosting_predictions)}")

# Plot predicted values vs actual target values
plt.figure(figsize=(10,6))
plt.scatter(y_test, gradient_boosting_predictions, color='blue', alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)
plt.title('Gradient Boosting: Predicted vs Actual Prices')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.grid(True)
plt.show()