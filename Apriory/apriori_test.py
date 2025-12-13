import os
import pandas as pd

from Apriori_model import generate_apriori_rules, recommend_by_sku

# =========================
# 1. Đánh giá 1 testcase
# =========================
def evaluate_single_testcase(file_path, df, rules=None, top_n=5, price_tol=10.0):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # SKU chính
    sku_main = lines[0].split("(")[1].split(")")[0]

    # SKU gợi ý cũ
    old_skus = []
    for line in lines:
        if line.startswith("- "):
            sku = line.split("|")[0].strip()[2:]
            old_skus.append(sku)

    # SKU gợi ý mới từ Apriori
    new_recs_df = recommend_by_sku(sku_main, df, rules=rules, top_n=top_n, price_tol=price_tol)
    new_skus = new_recs_df["SKU"].tolist()

    # So sánh: tất cả SKU cũ có trùng hết với gợi ý mới => đúng
    testcase_passed = set(old_skus) == set(new_skus)

    return {
        "file": os.path.basename(file_path),
        "sku_main": sku_main,
        "old_skus": old_skus,
        "new_skus": new_skus,
        "passed": testcase_passed
    }

# =========================
# 4. Đánh giá tất cả testcase trong folder
# =========================
def evaluate_all_testcases(df, output_dir, rules=None, top_n=5, price_tol=10.0):
    files = [f for f in os.listdir(output_dir) if f.endswith(".txt")]
    files.sort()
    results = []

    for f in files:
        file_path = os.path.join(output_dir, f)
        result = evaluate_single_testcase(file_path, df, rules=rules, top_n=top_n, price_tol=price_tol)
        results.append(result)
        # status = "PASSED ✅" if result["passed"] else "FAILED ❌"
        # print(f"{f}: {status}")

    results_df = pd.DataFrame(results)
    total_cases = len(results_df)
    passed_cases = results_df["passed"].sum()
    accuracy = passed_cases / total_cases * 100 if total_cases > 0 else 0

    print(f"\nTổng testcase: {total_cases}")
    print(f"Số testcase đúng: {passed_cases}/{total_cases}")
    print(f"Tỷ lệ chính xác: {accuracy:.2f}%")

    # Lưu kết quả chi tiết
    results_df.to_csv("apriori_testcase_results.csv", index=False)
    print("✔ Đã lưu kết quả chi tiết vào apriori_testcase_results.csv")
    return results_df

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    DATA_PATH = "Dataset/new_data_to_analysis.csv"
    OUTPUT_DIR = "Test/output_total"

    df, rules = generate_apriori_rules(
        DATA_PATH,
        min_support=0.1,
        min_lift=1
    )

    evaluate_all_testcases(
        df,
        OUTPUT_DIR,
        rules,
        top_n=5,
        price_tol=10.0
    )