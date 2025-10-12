import pandas as pd

df = pd.read_csv("Amazon Sale Report.csv")

# lay cac cot can thiet
df = df[["SKU", "Category", "Style", "Size", "Qty", "Amount", "Status"]]

# loai bo cac don hang khi chua giao thanh cong
df = df[df["Status"] == "Shipped"] 

# lay cac hang co qty = 1
df = df[df["Qty"] == 1]

# loai bo cac hang co gia tri null
df = df.dropna()

# loai bo cac hang co amount = 0
df = df[df["Amount"] > 0]

def split_sku(sku):
    parts = sku.split('-')
    style = parts[0] if parts else 'null'
    size = parts[-1] if len(parts) > 1 else 'null'
    middle = '-'.join(parts[1:-1]) if len(parts) > 2 else 'null'
    return pd.Series([style, middle, size])


df[['StyleCode', 'Core', 'SizeCode']] = df['SKU'].apply(split_sku)

df = df[["SKU", "Category", "Style", "Size", "Amount", "Core"]]

# Tao bang moi them vao thuoc tinh priceLevel
# Cac gia tri de chia values dua tren ham decribe()

values = [0, 457, 599, 790, 1139,2598]
label = ["low", "Medium", "High", "Premium", "Luxury"]

df["price_level"] = pd.cut(df["Amount"], bins=values, labels=label, include_lowest=True)

print(df.head())
print(df.shape)
print(df.info())
print(df.nunique())
print(df.describe())

df.to_csv("new_data_to_analysis.csv", index=False)





