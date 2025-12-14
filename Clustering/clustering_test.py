import os
import pandas as pd
from pathlib import Path

from Clutering_model import train_cluster, recommend_by_sku_cluster


# =========================
# 1. Đánh giá 1 testcase
# =========================
def evaluate_single_testcase(file_path, df, X, top_n=5, price_tol=10.0):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        return {
            "file": os.path.basename(file_path),
            "sku_main": None,
            "old_skus": [],
            "new_skus": [],
            "passed": False
        }

    # SKU chính
    sku_main = lines[0].split("(")[1].split(")")[0]

    # SKU gợi ý cũ
    old_skus = []
    for line in lines:
        if line.startswith("- "):
            sku = line.split("|")[0].strip()[2:]
            old_skus.append(sku)

    # SKU gợi ý mới từ clustering
    new_recs_df = recommend_by_sku_cluster(sku_main, df, X, top_n=top_n, price_tol=price_tol)
    new_skus = new_recs_df["SKU"].tolist() if not new_recs_df.empty else []

    testcase_passed = set(old_skus) == set(new_skus)

    return {
        "file": os.path.basename(file_path),
        "sku_main": sku_main,
        "old_skus": old_skus,
        "new_skus": new_skus,
        "passed": testcase_passed
    }

# =========================
# 2. Đánh giá tất cả testcase
# =========================
def evaluate_all_testcases(df, X, output_dir, top_n=5, price_tol=10.0):
    files = sorted([f for f in Path(output_dir).glob("*.txt")])
    results = []

    for f in files:
        result = evaluate_single_testcase(f, df, X, top_n=top_n, price_tol=price_tol)
        results.append(result)

    results_df = pd.DataFrame(results)
    total_cases = len(results_df)
    passed_cases = results_df["passed"].sum() if total_cases > 0 else 0
    accuracy = passed_cases / total_cases * 100 if total_cases > 0 else 0

    print(f"\nTổng testcase: {total_cases}")
    print(f"Số testcase đúng: {passed_cases}/{total_cases}")
    print(f"Tỷ lệ chính xác: {accuracy:.2f}%")

    # Lưu kết quả
    results_df.to_csv("Clustering/cluster_testcase_results.csv", index=False)
    print("Đã lưu kết quả chi tiết vào Clustering/cluster_testcase_results.csv")
    return results_df

# =========================
# MAIN
# =========================


if __name__ ==  "__main__":
    DATA_PATH = "Dataset/new_data_to_analysis.csv"
    OUTPUT_DIR = "Test/output_total"  # thư mục chứa các file output cũ

    df = pd.read_csv(DATA_PATH)

    df, X, kmeans, ct, sil_score, dbi_score = train_cluster(df, n_clusters=4)
    print(f"Clustering xong | Silhouette: {sil_score:.4f}, DBI: {dbi_score:.4f}")

    # Đánh giá tất cả testcase
    evaluate_all_testcases(df, X, OUTPUT_DIR, top_n=5, price_tol=10.0)
    print("Hoàn tất đánh giá tất cả testcase bằng clustering.")

