import pandas as pd
import matplotlib.pyplot as plt 

file_path = r"D:\qusay-alkaleh\study\maki̇ne-ogrenmesi\project\housing_price_dataset.csv"

# Veri setini yükleme
df = pd.read_csv(file_path)

# İlk 10 satırı yazdırma
print("### Veri Setinin İlk 10 Satırı ###")
print(df.head(10))
print("\n")


# Veri setinin boyutları
print("### Veri Setinin Boyutları ###")
print(f"Satır sayısı: {df.shape[0]}, Sütun sayısı: {df.shape[1]}")
print("\n")

# Veri seti bilgileri
print("### Veri Seti Bilgileri ###")
df.info()

# Histogram ile konut fiyatlarının dağılımını görselleştirme
plt.hist(df['price'],bins=20,edgecolor='black')
plt.title("Distribution of Housing Prices")
plt.xlabel("Fiyat")
plt.ylabel('Frekans')
plt.show()