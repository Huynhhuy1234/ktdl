import os
import pandas as pd

# =========================
# 1. Đọc dataset gốc
# =========================
df = pd.read_csv("new_data_to_analysis.csv")
df = df.drop_duplicates(subset="SKU", keep="first")

# Lấy 100 SKU đầu tiên (hoặc ít hơn nếu dataset nhỏ)
top_100 = df.head(100)

# Tạo thư mục output nếu chưa có
os.makedirs("outputs", exist_ok=True)

# =========================
# 2. Hàm gợi ý sản phẩm
# =========================
def get_recommendations(row, df_full, diff_limit=10):
    category = row["Category"]
    size = row["Size"]
    core = row["Core"]
    style = row["Style"]
    amount = row["Amount"]

    # --- Lọc sản phẩm cùng Category, Size, Core, khác Style ---
    filtered = df_full[
        (df_full["Category"] == category) &
        (df_full["Size"] == size) &
        (df_full["Core"] == core) &
        (df_full["Style"] != style)
    ].drop_duplicates(subset="SKU", keep="first")

    # --- Tính chênh lệch giá tuyệt đối ---
    filtered["amount_diff"] = abs(filtered["Amount"] - amount)

    # --- Lọc theo diff_limit ---
    filtered = filtered[filtered["amount_diff"] <= diff_limit]

    # --- Sắp xếp theo amount_diff tăng dần ---
    filtered = filtered.sort_values(by="amount_diff", ascending=True)

    # Lấy tối đa 5 sản phẩm
    return filtered.head(5)

# =========================
# 3. Tạo 100 file output
# =========================
for idx, (_, row) in enumerate(top_100.iterrows(), start=1):
    recs = get_recommendations(row, df, diff_limit=10)

    file_path = f"outputs/output_{idx}.txt"

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"=== PRODUCT #{idx} ({row['SKU']}) ===\n")
        f.write(f"Category: {row['Category']}\n")
        f.write(f"Size: {row['Size']}\n")
        f.write(f"Core: {row['Core']}\n")
        f.write(f"Price Level: {row['price_level']}\n")
        f.write(f"Style: {row['Style']}\n")
        f.write(f"Amount: {row['Amount']}\n\n")

        f.write("=== 5 GỢI Ý SẢN PHẨM ===\n")
        if len(recs) == 0:
            f.write("Không tìm thấy gợi ý phù hợp.\n")
        else:
            for _, r in recs.iterrows():
                f.write(f"- {r['SKU']} | {r['Style']} | {r['Amount']} VND | diff={r['amount_diff']}\n")

    print("Created:", file_path)

print("✔ DONE! Đã tạo xong 100 file output trong thư mục /outputs/")
