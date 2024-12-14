import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 

file_path = r"D:\qusay-alkaleh\study\maki̇ne-ogrenmesi\project\housing_price_dataset.csv"

# Veri setini yükleme
df = pd.read_csv(file_path)

# Veri setini yükle
df = pd.read_csv(file_path)

# Genel Bilgiler
print("### Veri Setine Genel Bakış ###")
print(f"Satır sayısı: {df.shape[0]}")
print(f"Sütun sayısı: {df.shape[1]}")
print("Sütun isimleri:", df.columns.tolist())
print("\n")

print("### Özet İstatistikler ###")
print(df.describe())
print("\n")

# Sayısal özelliklerin dağılımı
numerical_features = ['SquareFeet', 'Bedrooms', 'Bathrooms', 'YearBuilt', 'Price']
for feature in numerical_features:
    plt.figure(figsize=(10, 6))
    sns.histplot(df[feature], kde=True, bins=30, color='skyblue')
    plt.title(f'{feature} Dağılımı')
    plt.xlabel(feature)
    plt.ylabel('Frekans')
    plt.show()

# Aykırı değerler için kutu grafikler
for feature in numerical_features:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df[feature], color='orange')
    plt.title(f'{feature} Kutu Grafiği')
    plt.xlabel(feature)
    plt.show()

# Kategorik özelliklerin analizi
categorical_features = ['Neighborhood']
for feature in categorical_features:
    plt.figure(figsize=(10, 6))
    sns.countplot(x=df[feature], palette='viridis')
    plt.title(f'{feature} Sayım Grafiği')
    plt.xlabel(feature)
    plt.ylabel('Sayım')
    plt.xticks(rotation=45)
    plt.show()

# Fiyat ile diğer sayısal özellikler arasındaki ilişki
for feature in numerical_features:
    if feature != 'Price':
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=df[feature], y=df['Price'], hue=df['Neighborhood'], palette='viridis')
        plt.title(f'{feature} ile Fiyat Arasındaki İlişki')
        plt.xlabel(feature)
        plt.ylabel('Fiyat')
        plt.legend(title='Mahalle', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()

# Korelasyon ısı haritası
plt.figure(figsize=(10, 6))
corr = df[numerical_features].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Korelasyon Isı Haritası')
plt.show()