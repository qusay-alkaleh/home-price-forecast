import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Veri setini yükleme
file_path = r"D:\qusay-alkaleh\study\maki̇ne-ogrenmesi\project\housing_price_dataset.csv"
df = pd.read_csv(file_path)

# Kategorik sütunları seçme
df_label = df.select_dtypes(include='object')
print("### Kategorik sütunlar ###")
print(df_label.head())
print("\n")

# One-hot encoding yes/no columns
cat_colunms = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
df = pd.get_dummies(df, columns=cat_colunms, drop_first=True) 

# Kategorik sütunları etiketleme
df['furnishingstatus'] = LabelEncoder().fit_transform(df['furnishingstatus'])
print("### Etiketlenmiş kategorik sütunlar ###")
print(df.head())
print("\n")

# Etiketlenmiş verinin bilgi özetini yazdırma
print("### Etiketlenmiş verinin label-encoding sonrası bilgisi ###")
print(df.info())
print("\n")

# Birleştirilmiş veri çerçevesini kaydetme
save_path = r"D:\qusay-alkaleh\study\maki̇ne-ogrenmesi\project\dataset_preprocessed.csv"
df.to_csv(save_path, index=False)
