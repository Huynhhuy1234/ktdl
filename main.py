# main.py
import streamlit as st
import pandas as pd

from Apriori import Apriori_model
from Clustering import Clutering_model as Clustering
from UI.app import sidebar_selection, show_product_info

DATA_PATH = "Dataset/new_data_to_analysis.csv"

def main():
    st.set_page_config(page_title="Product Recommender System", layout="wide", page_icon="📊")
    
    # -----------------------
    # LOAD DATA
    # -----------------------
    @st.cache_data
    def load_dataset():
        try:
            df = pd.read_csv(DATA_PATH)
            return df
        except FileNotFoundError:
            st.error(f"Không tìm thấy file tại: {DATA_PATH}")
            st.stop()

    # -----------------------
    # TRAIN MODEL
    # -----------------------
    @st.cache_resource
    def train_models(df):
        df_apriori, rules = Apriori_model.generate_apriori_rules(DATA_PATH, min_support=0.1, min_lift=1)
        df_clustered, X, kmeans, ct, sil_score, dbi_score = Clustering.train_cluster(df, n_clusters=4)
        return rules, df_clustered, X, sil_score, dbi_score

    # -----------------------
    # RUN
    # -----------------------
    df_original = load_dataset()
    rules, df_clustered, X_features, sil_score, dbi_score = train_models(df_original)

    # SIDEBAR
    selected_sku = sidebar_selection(df_original)

    # SHOW PRODUCT INFO
    show_product_info(df_original, selected_sku)

    # -----------------------
    # OUTPUT: Apriori + Clustering
    # -----------------------
    st.title("📊 So sánh Thuật toán Gợi ý")
    st.markdown("---")
    col1, col2 = st.columns(2)

    # Apriori
    with col1:
        st.header("🔗 Apriori Recommendation")
        rec_apriori = Apriori_model.recommend_by_sku(selected_sku, df_original, rules=rules, top_n=5)
        if rec_apriori.empty:
            st.warning("Không tìm thấy gợi ý Apriori phù hợp.")
        else:
            for idx, row in rec_apriori.iterrows():
                st.markdown(f"**{row['SKU']}** | Cat: {row['Category']} | Core: {row['Core']} | Size: {row['Size']} | Price: ${row['Amount']}")

    # Clustering
    with col2:
        st.header("🎯 Clustering Recommendation")
        rec_cluster = Clustering.recommend_by_sku_cluster(selected_sku, df_clustered, X_features, top_n=5)
        if rec_cluster.empty:
            st.warning("Không tìm thấy gợi ý trong cụm.")
        else:
            for idx, row in rec_cluster.iterrows():
                st.markdown(f"**{row['SKU']}** | Cat: {row['Category']} | Style: {row['Style']} | Price: ${row['Amount']}")

    # -----------------------
    # METRICS
    # -----------------------
    st.markdown("---")
    st.subheader("📈 Hiệu năng Mô hình Global")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Silhouette Score", f"{sil_score:.4f}", delta="Càng cao càng tốt")
    m2.metric("Davies-Bouldin Index", f"{dbi_score:.4f}", delta="Càng thấp càng tốt", delta_color="inverse")
    m3.metric("Testcases Checked", "Total Files")
    m4.metric("Data Status", "Ready", "Live Analysis")

# ========================
# CHẠY APP
# ========================
if __name__ == "__main__":
    main()
