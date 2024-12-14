import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
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

# Initialize all the models
linear_model = LinearRegression()
svr_model = SVR()
decision_tree_model = DecisionTreeRegressor()
random_forest_model = RandomForestRegressor()
gradient_boosting_model = GradientBoostingRegressor()
knn_model = KNeighborsRegressor()

# Store the models in a list
models = [svr_model, decision_tree_model,
         random_forest_model, gradient_boosting_model, knn_model]

def evaluate_performance(y_true,y_pred):
    # Assuming y_true and y_pred are actual and predicted values
    mae = mt.mean_absolute_error(y_true, y_pred)
    mse = mt.mean_squared_error(y_true, y_pred)
    rmse = mt.root_mean_squared_error(y_true, y_pred)
    r2 = mt.r2_score(y_true, y_pred)

    print(f"MAE: {mae}")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"R-squared: {r2}")
    
    
def train_and_test_regression_models(models,X_train,y_train,X_test,y_test):
    for model in models:
        model_name = model.__class__.__name__
        model.fit(X_train, y_train)

        # Train performance
        print(f"Training Performance for {model_name}:")
        predicted_y_train = model.predict(X_train)
        evaluate_performance(y_train, predicted_y_train)

        # Test performance
        print(f"Testing Performance for {model_name}:")
        predicted_y_test = model.predict(X_test)
        evaluate_performance(y_test, predicted_y_test)

        # Visualization
        plt.figure(figsize=(10, 5))
        plt.scatter(y_test, predicted_y_test, color='blue', alpha=0.6, label='Predicted vs Actual')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal Fit')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f"{model_name} - Predicted vs Actual")
        plt.legend()
        plt.show()


train_and_test_regression_models(models,X_train,y_train,X_test,y_test)