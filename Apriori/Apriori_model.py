import os
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# =========================
# 1. Sinh rules Apriori
# =========================
def generate_apriori_rules(csv_path, min_support=0.1, min_lift=1):
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates(subset="SKU", keep="first")

    cols = ["Category", "Style", "Size", "Core", "price_level"]
    df_hot = pd.get_dummies(df[cols])
    
    frequent_itemsets = apriori(df_hot, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_lift)
    rules = rules.sort_values(by="lift", ascending=False)
    return df, rules

# =========================
# 2. Gợi ý sản phẩm cho SKU
# =========================
def recommend_by_sku(sku, df, rules=None, top_n=5, price_tol=10.0):
    product = df[df["SKU"] == sku]
    if product.empty:
        return pd.DataFrame()
    product = product.iloc[0]

    category = product["Category"]
    core = product["Core"]
    size = product["Size"]
    price_level = product["price_level"]
    amount = product["Amount"]

    candidates = df[
        (df["SKU"] != sku) &
        (df["Size"] == size) &
        (df['Core'] == core) &
        (df['Category'] == category)
    ].copy()

    candidates["similarity_score"] = (
        (candidates["Category"] == category).astype(int) +
        (candidates["Core"] == core).astype(int) +
        (candidates["Size"] == size).astype(int) +
        (candidates["price_level"] == price_level).astype(int)
    )

    candidates = candidates[candidates["similarity_score"] >= 2]
    candidates["amount_diff"] = abs(candidates["Amount"] - amount)
    candidates = candidates[candidates["amount_diff"] <= price_tol]
    candidates = candidates.drop_duplicates(subset="SKU")
    candidates = candidates.sort_values(by=["similarity_score","amount_diff"], ascending=[False, True])

    return candidates.head(top_n)[
        ["SKU","Category","Core","Size","Amount","similarity_score","amount_diff"]
    ]

