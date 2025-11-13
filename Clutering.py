import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_distances

DATA_PATH = "new_data_to_analysis.csv"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

def train_cluster(k=8):
    df = pd.read_csv(DATA_PATH).drop_duplicates(subset=["SKU"]).reset_index(drop=True)
    X, ct = build_features(df)
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=1024)
    labels = kmeans.fit_predict(X)
    sil = silhouette_score(X, labels) if k > 1 else np.nan
    dbi = davies_bouldin_score(X, labels) if k > 1 else np.nan
    df["Cluster"] = labels
    return df, X, kmeans, ct, sil, dbi

def recommend_by_sku_cluster(sku, top_n=5, price_tol=10.0):
    df, X, model, ct, sil, dbi = train_cluster(k=8)
    row = df[df["SKU"] == sku]
    if row.empty:
        print("⚠️ SKU không tồn tại.")
        return pd.DataFrame()
    idx = row.index[0]
    cluster = int(row["Cluster"].iloc[0])
    ref_amount = float(row["Amount"].iloc[0]) if "Amount" in df.columns else None

    cand = df[(df["Cluster"] == cluster) & (df["SKU"] != sku)].copy()
    if cand.empty:
        return pd.DataFrame()

    # Tính khoảng cách cosine trong không gian feature
    ref_vec = X[idx:idx+1]
    cand_vecs = X[cand.index]
    cand["distance"] = cosine_distances(ref_vec, cand_vecs).ravel()

    # Ưu tiên mềm: cùng Core/Size/Category
    def priority(r):
        score = 0.0
        if "Core" in df.columns and r["Core"] == row["Core"].iloc[0]:
            score += 2.0
        if "Size" in df.columns and r["Size"] == row["Size"].iloc[0]:
            score += 1.5
        if "Category" in df.columns and r["Category"] == row["Category"].iloc[0]:
            score += 1.2
        return score
    cand["priority"] = cand.apply(priority, axis=1)

    # Lọc theo chênh lệch giá
    if ref_amount is not None:
        cand = cand[(cand["Amount"].notna()) & (cand["Amount"] - ref_amount).abs() <= price_tol]

    cand = cand.sort_values(["priority","distance"], ascending=[False, True])
    return cand.head(top_n)[["SKU","Category","Style","Size","Core","Amount","Cluster","priority","distance"]]

if __name__ == "__main__":
    sku_input = "JNE2270-KR-487-A-M"
    print(f"=== GỢI Ý (Cluster) CHO {sku_input} ===")
    recs_cluster = recommend_by_sku_cluster(sku_input, top_n=5)
    print(recs_cluster)
    out_path = OUTPUT_DIR / f"cluster_recs_{sku_input}.csv"
    recs_cluster.to_csv(out_path, index=False)
    print(f"📄 Đã lưu gợi ý Clustering vào: {out_path.resolve().as_posix()}")
