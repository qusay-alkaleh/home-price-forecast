import pandas as pd
from scipy.stats import zscore
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Veri setini yükleme
file_path = r"D:\qusay-alkaleh\study\maki̇ne-ogrenmesi\project\housing_price_dataset.csv"
df = pd.read_csv(file_path)
numeric_cols = ["price", "area", "bedrooms", "bathrooms", "stories", "parking"]
categorical_cols = ["mainroad", "guestroom", "hotwaterheating", "airconditioning"]

# Drop unnecessary columns & replace '?' or similar placeholders if present
df.drop(columns=["prefarea", "furnishingstatus", "basement"], inplace=True)
print("### After Dropping Unnecessary Columns ###")
print(df.head())
df.replace('?', None, inplace=True)
print("\n")

# Impute missing values
numeric_imputer = SimpleImputer(strategy='mean')  # For numeric columns
categorical_imputer = SimpleImputer(strategy='most_frequent')  # For categorical columns

df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])
df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])
print("### After Handling Missing Values ###")
print(df.isnull().sum())
print("\n")

# Apply z-scores for numeric columns
print("### Appling z-score on Dataset ###")
z_scores = df[numeric_cols].apply(zscore)
threshold = 3.0
outliers = (z_scores.abs() > threshold).any(axis=1)
df_cleaned = df[~outliers]
print(df_cleaned)
print("\n")

# Label encoding for categorical variables
label_encoder = LabelEncoder()
for col in categorical_cols:
    df_cleaned[col] = label_encoder.fit_transform(df_cleaned[col])

print("### After Label Encoding ###")
print(df_cleaned.head())
print("\n")

# Apply scaling on dataset using MinMaxScaler (normalization)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df_cleaned)
scaled_df = pd.DataFrame(scaled_data, columns=df_cleaned.columns)

print("### After Scalering Using Normalization ###")
print(scaled_df.describe().round(3))
print("\n")






