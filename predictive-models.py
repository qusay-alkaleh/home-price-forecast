import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import matplotlib.pyplot as plt 

file_path = r"D:\qusay-alkaleh\study\maki̇ne-ogrenmesi\project\labeled_dataset.csv"
df = pd.read_csv(file_path)

X = df.drop('price' , axis=1)
y = df['price']

X_train , X_test , y_train , y_test = train_test_split(X , y, 
                                                       test_size=0.2, 
                                                       random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

xgboost = XGBRegressor()


paramm = {
    'n_estimators': [20000],
    'max_depth': [1],
    'gamma': [0.008],
    'reg_lambda': [0.05],
    'reg_alpha': [0.05],
    'subsample': [0.5],
    'colsample_bytree': [0.6],
    'min_samples_split': [0.01],
    'min_samples_leaf': [0.1],
    'learning_rate' : [0.03]
}

grd = GridSearchCV(xgboost , paramm , cv=10 , n_jobs=-1 ,verbose=1)
grd.fit(X_train_scaled , y_train)

y_pred_grd = grd.predict(X_test_scaled)
r2 = grd.score(X_test_scaled , y_test)
mean_squ = mean_squared_error(y_test , y_pred_grd)**5
print(mean_squ)
print(r2)

plt.scatter(y_test, y_pred_grd)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")

# Plot a line showing perfect prediction
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', label='Perfect Prediction')

# Calculate and plot a regression line
m, b = np.polyfit(y_test, y_pred_grd, 1)
plt.plot(y_test, m*y_test + b, color='coral', label='Regression Line')

plt.legend()
plt.show()