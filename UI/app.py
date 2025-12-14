# app_functions.py
import streamlit as st
import pandas as pd

# ==========================================
# HIỂN THỊ THÔNG TIN SẢN PHẨM
# ==========================================
def show_product_info(df, selected_sku):
    current_product = df[df["SKU"] == selected_sku].iloc[0]
    st.subheader("📦 Thông tin chi tiết")
    st.markdown(f"**SKU:** `{current_product['SKU']}`")
    st.markdown(f"**Cat:** {current_product['Category']} | **Style:** {current_product['Style']}")
    st.markdown(f"**Size:** {current_product['Size']} | **Core:** {current_product['Core']}")
    st.markdown(f"**Price:** ${current_product['Amount']}")

# ==========================================
# SIDEBAR: Lọc giá và chọn SKU
# ==========================================
def sidebar_selection(df):
    st.sidebar.title("🎛️ Control Panel")
    st.sidebar.divider()
    
    # BỘ LỌC GIÁ
    min_price = int(df["Amount"].min())
    max_price = int(df["Amount"].max())
    
    price_range = st.sidebar.slider(
        "Khoảng giá mong muốn:",
        min_value=min_price, max_value=max_price,
        value=(min_price, max_price), step=50
    )
    
    filtered_df = df[
        (df["Amount"] >= price_range[0]) & 
        (df["Amount"] <= price_range[1])
    ]
    filtered_skus = filtered_df["SKU"].unique()
    st.sidebar.caption(f"Tìm thấy **{len(filtered_skus)}** sản phẩm.")
    
    if len(filtered_skus) == 0:
        st.sidebar.error("Không có sản phẩm nào!")
        st.stop()
    
    selected_sku = st.sidebar.selectbox("🔍 Chọn hoặc nhập SKU:", filtered_skus, index=0)
    return selected_sku
