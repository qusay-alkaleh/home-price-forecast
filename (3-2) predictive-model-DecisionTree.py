import pandas as pd
from sklearn.tree import DecisionTreeRegressor
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

# Decision Tree model
decision_tree_model = DecisionTreeRegressor()
decision_tree_model.fit(X_train, y_train)

# Predictions for Decision Tree model
decision_tree_predictions = decision_tree_model.predict(X_test)

# Evaluate Decision Tree model
print("\nDecision Tree Performance:")
print(f"traing score of decision tree {decision_tree_model.score(X_train, y_train)}")
print(f"R^2 Score: {r2_score(y_test, decision_tree_predictions)}")
print(f"Mean Squared Error: {mean_squared_error(y_test, decision_tree_predictions)}")

# Plot predicted values vs actual target values
plt.figure(figsize=(10,6))
plt.scatter(y_test, decision_tree_predictions, color='blue', alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)
plt.title('Decision Treet: Predicted vs Actual Prices')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.grid(True)
plt.show()