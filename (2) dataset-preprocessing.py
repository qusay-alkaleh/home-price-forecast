import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Veri setini yükleme
file_path = r"D:\qusay-alkaleh\study\maki̇ne-ogrenmesi\project\housing_price_dataset.csv"
df = pd.read_csv(file_path)

# Boş değerlerin kontrolü ve açıklaması
print("### Boş Değerler ###")
missing_values = df.isnull().sum()
missing_percentage = (missing_values / df.shape[0]) * 100

if missing_percentage[missing_percentage > 0].empty:
    print("Her sütunda boş değer yok (Boş değer yüzdesi: 0%)")
else:
    print(f"Her sütundaki boş değerlerin yüzdesi:\n{missing_percentage[missing_percentage > 0]}%")
print("\n")

# Kategorik ve Sayısal Sütunları Ayrıştırma
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
categorical_columns = df.select_dtypes(include=['object']).columns

print("### Kategorik Sütunlar ###")
print(categorical_columns)
print("\n")

print("### Sayısal Sütunlar ###")
print(numerical_columns)
print("\n")

# Sayısal özellikler için boş değerlerin görseli
for column in numerical_columns:
    if df[column].isnull().sum() > 0:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[column], kde=True, bins=30, color='skyblue')
        plt.title(f'{column} Dağılımı (Boş Değerler Var)')
        plt.xlabel(column)
        plt.ylabel('Frekans')
        plt.show()

# Kategorik özelliklerdeki boş değerlerin sayısı
for column in categorical_columns:
    if df[column].isnull().sum() > 0:
        print(f"{column} sütunundaki boş değer sayısı: {df[column].isnull().sum()}")
        
# Genel istatistikleri göster
print("### Veri Seti Özeti ###")
print(df.describe())
print("\n")

# Label Encoding Uygulama
label_encoder = LabelEncoder()

# Kategorik sütunlarda Label Encoding işlemi
for column in categorical_columns:
    df[column] = label_encoder.fit_transform(df[column])
    
print("### Etiket Kodları Uygulandıktan Sonra Veri Seti ###")
print(df.head())
print("\n")

save_labeled_path = r"D:\qusay-alkaleh\study\maki̇ne-ogrenmesi\project\dataset_labeled.csv"
df.to_csv(save_labeled_path, index=False)

# Özellikler (X) ve hedef (y) sütunlarını ayır
X = df.drop(columns=["Price"])
y = df["Price"]

# Veriyi eğitim ve test olarak ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# MinMaxScaler uygulama
scaler = MinMaxScaler()

# Sadece eğitim verisini kullanarak ölçekleme (fit_transform)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("### Ölçeklendirilmiş Eğitim Verisi (İlk 5 Satır) ###")
print(X_train_scaled[:5])


# Initialize PCA with an RBF kernel
pca = PCA(n_components=5)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Check the resulting components
print("Transformed Training Data Shape:", X_train_pca.shape)

# Train a model
model = LinearRegression()
model.fit(X_train_pca, y_train)

# Predict on training data
y_train_pred = model.predict(X_train_pca)
train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

# Predict on test data
y_test_pred = model.predict(X_test_pca)
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("### Training Performance ###")
print(f"Mean Squared Error: {train_mse}")
print(f"R^2 Score: {train_r2}")

print("\n### Test Performance ###")
print(f"Mean Squared Error: {test_mse}")
print(f"R^2 Score: {test_r2}")

# Convert PCA transformed data back to DataFrame
X_train_pca_df = pd.DataFrame(X_train_pca, columns=[f"PC{i+1}" for i in range(X_train_pca.shape[1])])
X_test_pca_df = pd.DataFrame(X_test_pca, columns=[f"PC{i+1}" for i in range(X_test_pca.shape[1])])

# Reattach the target variable to the PCA-transformed data
train_dataset = X_train_pca_df.copy()
train_dataset["Price"] = y_train.reset_index(drop=True)

test_dataset = X_test_pca_df.copy()
test_dataset["Price"] = y_test.reset_index(drop=True)

# Combine train and test datasets
dataset_scaled = pd.concat([train_dataset, test_dataset], axis=0)
save_scaled_path = r"D:\qusay-alkaleh\study\maki̇ne-ogrenmesi\project\dataset_scaled.csv"
dataset_scaled.to_csv(save_scaled_path, index=False)