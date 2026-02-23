import streamlit as st
st.set_page_config(
    page_title="Fundamental Analisis Data", 
    layout="wide"
)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from babel.numbers import format_currency

sns.set_style("darkgrid")

# PENGOLAHAN DATA ----------
## sales data
@st.cache_data
def load_sales_data(data_path):
    sales_data_df = pd.read_csv(data_path)

    datetime_columns = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
        "shipping_limit_date"
    ]

    for col in datetime_columns:
        sales_data_df[col] = pd.to_datetime(sales_data_df[col], errors='coerce')

    sales_data_df.sort_values(by='order_purchase_timestamp', inplace=True)

    return sales_data_df

## users data
@st.cache_data
def load_users_data(customers_path, sellers_path):
    customers_df = pd.read_csv(customers_path)
    sellers_df = pd.read_csv(sellers_path)

    return customers_df, sellers_df

## Load data
sales_data_df = load_sales_data('sales_data.csv')
customers_df, sellers_df = load_users_data('customers_geo.csv', 'sellers_geo.csv')

# DASHBOARD UI ----------
st.markdown(
    "<h1 style='text-align: center;'>Dashboard Penjualan E-Commerce</h1>", 
    unsafe_allow_html=True
)

# FILTERING DATA ----------
## Komponen filter waktu
min_date = sales_data_df['order_purchase_timestamp'].min().date()
max_date = sales_data_df['order_purchase_timestamp'].max().date()

## Top Bar Filter
st.markdown(
    """
    <style> 
    .top-bar {
        padding: 0.5rem 0 1.2rem 0;
    }
    
    .logo-wrapper {
        display: flex;
        align-items: center;
    }
    
    .logo-wrapper img {
        display: flex;
        align-items: flex-end;
        width: 100%;
        height: auto;
    }

    .date-filter {
        display: flex;
        align-items: flex-end;
        gap: 1rem;
    }

    .date-filter > div {
        display: flex;
        flex-direction: column;
    }
    </style>
    """, unsafe_allow_html=True
)

with st.container():
    col_logo, col_space_top_bar, col_filter = st.columns([1, 2, 3])

    # Logo
    with col_logo:
        st.markdown(
            """
            <div class="logo-wrapper">
                <img src="https://github.com/Sulbae/Fundamental-Analisis-Data/blob/4d6e16f7fbc32faa7581a9f81ac7d9c87a7c18f4/assets/pngtree-shopping-bag-icon.png?raw=true">
            </div>
            """,
            unsafe_allow_html=True
        )

    # Space Top Bar
    with col_space_top_bar:
        st.markdown(" ")
    
    # Filter
    with col_filter:
        st.markdown("### Pilih Periode Penjualan")

        st.markdown('<div class="date-filter">', unsafe_allow_html=True)

        time1, time2 = st.columns(2)

        with time1:
            start_date = st.date_input(
                "Start Date", 
                min_value=min_date, max_value=max_date, value=min_date
            )

        with time2:
            end_date = st.date_input(
                "End Date", 
                min_value=min_date, max_value=max_date, value=max_date
            )
        
        st.markdown('</div>', unsafe_allow_html=True)

        if start_date > end_date:
            st.error("Start Date tidak boleh lebih besar dari End Date!")
            st.stop()
    
st.markdown("---")

## Simpan data terfilter yang akan digunakan
filtered_df = sales_data_df[
    (sales_data_df['order_purchase_timestamp'].dt.date >= start_date) & 
    (sales_data_df['order_purchase_timestamp'].dt.date <= end_date)
].copy()


# HELPER FUNCTIONS ----------
## Fungsi untuk membuat DataFrame tren penjualan bulanan
def create_monthly_orders_df(df):
    monthly_orders_df = (
        df.resample(rule='M', on='order_purchase_timestamp')
        .agg({
            'order_id': 'nunique',
            'payment_value': 'sum'
        })
        .reset_index()
    )

    monthly_orders_df.rename(columns={
        'order_id': 'total_orders',
        'payment_value': 'total_sales'
    }, inplace=True)

    monthly_orders_df['order_purchase_timestamp'] = monthly_orders_df['order_purchase_timestamp'].dt.strftime('%m-%Y')
    
    return monthly_orders_df

## Visualisasi tren penjualan bulanan
def sales_trend_viz(x, y):
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(x, y, marker='o', linewidth=2, markersize=8, color="#90CAF9")
    ax.set_title('Tren Penjualan Bulanan', fontsize=16, loc="center")
    ax.set_xlabel('Bulan')
    ax.set_ylabel('Total Penjualan')

    plt.xticks(rotation=45)
    
    st.pyplot(fig)
    plt.close(fig)

## Visualisasi penjualan produk
def plot_product_sales(data_df, ascending=False, title=""):
    data = (
        data_df
        .rename(columns={'product_category_name_english': 'product_category'})
        .groupby('product_category')
        .size()
        .reset_index(name='quantity')
        .sort_values(by='quantity', ascending=ascending)
        .head(5)
    )

    fig, ax = plt.subplots(
        figsize=(10, 4),
        facecolor='none'   
    )
    
    colors = ["#90CAF9", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]

    sns.barplot(
        data=data,
        x='quantity',
        y='product_category',
        palette=colors,
        ax=ax
    )

    ax.set_title(title, fontsize=14, color="white")
    ax.set_xlabel('Jumlah Terjual', fontsize=12, color="white")
    ax.set_ylabel('Kategori Produk', fontsize=12, color="white")
    
    st.pyplot(fig)
    plt.close(fig)

## Peta distribusi lokasi users
'''
def create_users_map_df(customers_df, sellers_df):
    # Base map
    m = folium.Map(
        location=[-14.2, -51.9], 
        zoom_start=4,
        tiles='cartodbpositron'
    )

    ## Customer Layer
    customer_layer = folium.FeatureGroup(name='Customers')

    HeatMap(
        data=list(zip(customers_df['geolocation_lat'], customers_df['geolocation_lng'])),
        radius=10,
        blur=15,
        min_opacity=0.4,
        gradient={0.0: 'gray', 1.0: 'blue'} # warna gray = min, blue = max
    ).add_to(customer_layer)

    customer_layer.add_to(m)

    ## Seller Layer
    seller_layer = folium.FeatureGroup(name='Sellers')

    for idx, row in sellers_df.iterrows():
        folium.CircleMarker(
            location=[row['geolocation_lat'], row['geolocation_lng']],
            radius=4,
            color='#FFA500',
            fill=True,
            fill_color='#FFA500',
            fill_opacity=0.7,
            popup="Seller"
        ).add_to(seller_layer)

    seller_layer.add_to(m)

    # Layer Control
    folium.LayerControl(collapsed=False).add_to(m)

    return m
'''

# VISUALISASI CHART ----------
## Buat DataFrame untuk analisis tren penjualan bulanan
with st.container():
    st.subheader("Ringkasan Transaksi")
    ## Layout untuk menampilkan metrik
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    with kpi1:
        total_sales = filtered_df['payment_value'].sum()
        st.metric(label="Total Sales", value=format_currency(total_sales, 'BRL', locale='pt_BR'))

    with kpi2:
        avg_sales = filtered_df['payment_value'].mean()
        st.metric(label="Avg. Sales", value=format_currency(avg_sales, 'BRL', locale='pt_BR'))

    with kpi3:
        total_orders = filtered_df['order_id'].nunique()
        st.metric(label="Total Orders", value=total_orders)

    with kpi4:
        num_customer = filtered_df['customer_id'].nunique()
        st.metric(label="Total Customers", value=num_customer)

with st.container():
    st.subheader("Kinerja Layanan")
    ## Layout untuk menampilkan metrik
    kpi5, kpi6, kpi7, kpi8 = st.columns(4)

    with kpi5:
        popular_payment = filtered_df['payment_type'].value_counts().index[0]
        st.metric(label="Popular Payment Type", value=popular_payment)

    with kpi6:
        unpopular_payment = filtered_df['payment_type'].value_counts().index[-1]
        st.metric(label="Unpopular Payment Type", value=unpopular_payment)

    with kpi7:
        delivery_success_rate = (filtered_df['order_status'] == 'delivered').mean() * 100
        st.metric(label="Delivery Success Rate", value=f"{delivery_success_rate:.2f}%")

    with kpi8:
        avg_review = filtered_df['review_score'].mean()
        st.metric(label="Avg. Review Score", value=round(avg_review, 2))

## Tampilkan chart tren penjualan bulanan
st.subheader("📈 Tren Penjualan")
monthly_sales_df = create_monthly_orders_df(filtered_df)
sales_trend_viz(monthly_sales_df['order_purchase_timestamp'], monthly_sales_df['total_sales'])

## Tampilkan chart produk terlaris dan terburuk
col1, col2 = st.columns(2)

with col1:
    plot_product_sales(
        data_df=filtered_df,
        ascending=False,
        title="Produk Terlaris"
    )

with col2:
    plot_product_sales(
        data_df=filtered_df,
        ascending=True,
        title="Produk Penjualan Terendah"
    )

## Tampilkan peta distribusi users
## st.subheader("Distribusi Lokasi Users")
## st_folium(create_users_map_df(customers_df, sellers_df), height=650)

st.caption("Copyright © 2026 - Data Science")