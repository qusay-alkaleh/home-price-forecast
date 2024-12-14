import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

file_path = r"D:\qusay-alkaleh\study\maki̇ne-ogrenmesi\project\dataset_labeled.csv"
df = pd.read_csv(file_path)

X = df.drop('Price' , axis=1)
y = df['Price']

y = pd.cut(y, bins=3, labels=["Low", "Medium", "High"])

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    train_size=0.75, 
                                                    random_state=42)

# Naive Bayes model
naive_bayes_model = GaussianNB()
naive_bayes_model.fit(X_train, y_train)

# Predictions for Naive Bayes model
naive_bayes_predictions = naive_bayes_model.predict(X_test)

# Evaluate Naive Bayes model
print("Naive Bayes Performance:")
print(f"Accuracy: {naive_bayes_model.score(X_test, y_test)}")

# Plot predicted values vs actual target values
plt.figure(figsize=(10,6))
plt.scatter(y_test, naive_bayes_predictions, color='blue', alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)
plt.title('GaussianNB: Predicted vs Actual Prices')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.grid(True)
plt.show()