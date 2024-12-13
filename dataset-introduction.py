import pandas as pd
import seaborn as sns
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

# 'area' ve 'price' arasındaki ilişkiyi incelemek için yan yana saçılım ve çizgi grafikleri oluşturma
fig, axes = plt.subplots(1, 2, figsize=(8, 3))
sns.scatterplot(x='area', y='price', data=df, ax=axes[0])
axes[0].set_title('Saçılım Grafiği: Alan vs Fiyat')
sns.lineplot(x='area', y='price', data=df, ax=axes[1])
axes[1].set_title('Çizgi Grafiği: Alan vs Fiyat')
plt.tight_layout()
plt.show()

# Farklı kategorik özellikler için 'price' dağılımını görselleştirmek amacıyla kutu grafikleri oluşturma
fig, axes = plt.subplots(2, 3, figsize=(10, 7))

# Analiz edilecek kategorik özellikler listesi
box_cat = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
axes = axes.flatten()
for i, feature in enumerate(box_cat):
    sns.boxplot(x=feature, y='price', data=df, ax=axes[i])
    axes[i].set_title(f'Kutu Grafiği: {feature} vs Fiyat')
plt.tight_layout()
plt.show()

# 'furnishingstatus' dağılımını göstermek için pasta grafiği oluşturma
plt.figure(figsize=(6, 6))
plt.pie(df['furnishingstatus'].value_counts(), 
        labels=df['furnishingstatus'].unique(), 
        autopct='%1.1f%%',
        startangle=90)
plt.title('Eşya Durumu Dağılımı')
plt.show()

# 'furnishingstatus' ve 'parking' kategorilerine göre ortalama fiyatı göstermek için çubuk grafik oluşturma
avg_price = df.groupby(['furnishingstatus', 'parking'])['price'].mean().unstack()

avg_price.plot(kind='bar', figsize=(8, 5), colormap='viridis')
plt.xlabel('Eşya Durumu')
plt.ylabel('Ortalama Fiyat')
plt.title('Eşya Durumu ve Otopark Göre Ortalama Fiyat')
plt.xticks(rotation=0)
plt.legend(title='Otopark')
plt.tight_layout()
plt.show()