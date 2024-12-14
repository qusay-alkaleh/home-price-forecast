import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

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

save_labeled_path = r"D:\qusay-alkaleh\study\maki̇ne-ogrenmesi\project\dataset_labeled.csv"
df.to_csv(save_labeled_path, index=False)


