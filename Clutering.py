import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity

# =================================
# Cấu hình đường dẫn
# =================================
DATA_PATH = "new_data_to_analysis.csv"
OUTPUT_DIR = Path("outputs_cluster")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# =================================
# 1. Build feature matrix
# =================================
def build_features(df: pd.DataFrame):
    cat_cols = [c for c in ["Category","Style","Size","Core"] if c in df.columns]
    num_cols = [c for c in ["Amount"] if c in df.columns]
    ct = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", StandardScaler(), num_cols),
        ],
        remainder="drop",
    )
    X = ct.fit_transform(df[cat_cols + num_cols])
    return X, ct

# =================================
# 2. Train MiniBatchKMeans
# =================================
def train_cluster(df: pd.DataFrame, n_clusters=8, batch_size=1024, random_state=42):
    df = df.drop_duplicates(subset=["SKU"]).reset_index(drop=True)
    X, ct = build_features(df)
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, random_state=random_state)
    labels = kmeans.fit_predict(X)
    df["Cluster"] = labels
    sil_score = silhouette_score(X, labels) if n_clusters > 1 else np.nan
    dbi_score = davies_bouldin_score(X, labels) if n_clusters > 1 else np.nan
    return df, X, kmeans, ct, sil_score, dbi_score

# =================================
# 3. Recommend SKU
# =================================
def recommend_by_sku_cluster(sku, df, X, top_n=5, price_tol=10.0):
    row = df[df["SKU"] == sku]
    if row.empty:
        return pd.DataFrame()
    row = row.iloc[0]
    cluster = row["Cluster"]
    ref_amount = row["Amount"]

    # Candidate cùng cluster, khác SKU
    cand = df[(df["Cluster"] == cluster) & (df["SKU"] != sku)].copy()
    if cand.empty:
        return pd.DataFrame()

    # Lọc theo Category, Size, Core
    cand = cand[
        (cand["Category"] == row["Category"]) &
        (cand["Size"] == row["Size"]) &
        (cand["Core"] == row["Core"])
    ]
    if cand.empty:
        return pd.DataFrame()

    # Lọc ± Amount
    cand = cand[(cand["Amount"] >= ref_amount - price_tol) & (cand["Amount"] <= ref_amount + price_tol)]
    if cand.empty:
        return pd.DataFrame()

    # Tính cosine similarity
    ref_vec = X[row.name:row.name+1]
    cand_vecs = X[cand.index]
    cand["cosine_sim"] = cosine_similarity(ref_vec, cand_vecs).ravel()

    # Khoảng cách Amount
    cand["amount_diff"] = abs(cand["Amount"] - ref_amount)

    # Sắp xếp: cosine cao → amount_diff thấp
    cand = cand.sort_values(by=["cosine_sim","amount_diff"], ascending=[False, True])

    return cand.head(top_n)[["SKU","Category","Core","Style","Size","Amount","Cluster","cosine_sim","amount_diff"]]

# =================================
# 4. Write recommendations + metrics tổng hợp
# =================================
def write_recommendations(sku_list, df, X, top_n=5, price_tol=10.0):
    OUTPUT_DIR.mkdir(exist_ok=True)
    metrics_records = []

    for idx, sku in enumerate(sku_list, start=1):
        recs = recommend_by_sku_cluster(sku, df, X, top_n=top_n, price_tol=price_tol)
        file_path = OUTPUT_DIR / f"output_{idx}.txt"
        # Lưu file text
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"=== GỢI Ý SẢN PHẨM CHO SKU: {sku} ===\n\n")
            if recs.empty:
                f.write("Không tìm thấy sản phẩm gợi ý phù hợp.\n")
            else:
                for _, r in recs.iterrows():
                    f.write(f"- {r['SKU']} | {r['Category']} | {r['Core']} | {r['Style']} | {r['Size']} | "
                            f"{r['Amount']} | cosine={r['cosine_sim']:.4f} | diff={r['amount_diff']:.2f}\n")
        print(f"✔ Đã lưu gợi ý SKU {sku} -> {file_path}")

        # Lưu metrics tổng hợp cho SKU
        metrics_records.append({
            "SKU": sku,
            "num_candidates": len(df[(df["Cluster"] == df.loc[df["SKU"]==sku].index[0]) & 
                                      (df["SKU"] != sku)]),
            "num_after_filter": len(recs),
            "max_cosine": recs["cosine_sim"].max() if not recs.empty else np.nan,
            "mean_cosine": recs["cosine_sim"].mean() if not recs.empty else np.nan,
        })

    # Lưu metrics tổng hợp vào 1 file CSV
    metrics_df = pd.DataFrame(metrics_records)
    metrics_file = OUTPUT_DIR / "recommendation_metrics_summary.csv"
    metrics_df.to_csv(metrics_file, index=False)
    print(f"📄 Metrics tổng hợp gợi ý đã lưu tại {metrics_file}")

# =================================
# 5. Save clustering metrics
# =================================
def save_clustering_metrics(sil, dbi, filename=OUTPUT_DIR / "clustering_metrics.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"Silhouette Score: {sil}\n")
        f.write(f"Davies-Bouldin Index: {dbi}\n")
    print(f"📄 Metrics clustering đã lưu tại {filename}")

# =================================
# MAIN
# =================================
if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    df, X, kmeans, ct, sil_score, dbi_score = train_cluster(df, n_clusters=8)
    print(f"✅ Clustering xong | Silhouette: {sil_score:.4f}, DBI: {dbi_score:.4f}")

    save_clustering_metrics(sil_score, dbi_score)

    # Lấy 100 SKU đầu tiên để test
    sku_list = df["SKU"].unique()[:100]
    write_recommendations(sku_list, df, X, top_n=5, price_tol=10)
    print("🎉 Hoàn tất: tất cả 100 file gợi ý đã tạo xong.")
