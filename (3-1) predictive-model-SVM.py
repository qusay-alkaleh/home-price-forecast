import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import sklearn.metrics as mt
import matplotlib.pyplot as plt

file_path = r"D:\qusay-alkaleh\study\maki̇ne-ogrenmesi\project\dataset_scaled.csv"
df = pd.read_csv(file_path)

X = df.drop('Price' , axis=1)
y = df['Price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Initialize and train the SVR model
svr_model = SVR()
svr_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svr_model.predict(X_test)

# Plot the predicted values against the actual target values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', label='Predicted')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Perfect Prediction')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('SVR: Predicted vs Actual Price')
plt.legend()
plt.show()

# Optionally, evaluate the model using metrics
print("Mean Squared Error:", mt.mean_squared_error(y_test, y_pred))
print("R-squared:", mt.r2_score(y_test, y_pred))