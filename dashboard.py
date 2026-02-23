import streamlit as st
st.set_page_config(
    page_title="Fundamental Analisis Data", 
    layout="wide"
)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
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
st.title("Dashboard Penjualan E-Commerce")

# FILTERING DATA ----------
## Komponen filter waktu
min_date = sales_data_df['order_purchase_timestamp'].min().date()
max_date = sales_data_df['order_purchase_timestamp'].max().date()

## Top Bar Filter
with st.container():
    col_logo, col_filter = st.columns([1, 5])

    # Logo
    with col_logo:
        st.image(
            "",
            height=50
        )

    # Filter
    with col_filter:
        st.markdown("### Pilih Periode Penjualan")

        time1, time2, apply_f = st.columns([2, 2, 1])

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

        with apply_f:
            st.markdown(" ")
            if st.button("Apply Filter"):
                if start_date > end_date:
                    st.error("Start Date harus sebelum End Date!")
                else:
                    st.success(f"Filter berhasil diterapkan")

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
        .agg(total_orders=('order_id', 'nunique'),
             total_sales=('sales', 'sum'))
        .reset_index()
    )

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
def plot_product_sales(data_df, column_y, column_x, ascending=False, title=""):
    data = (
        data_df.groupby(column_y)[column_x]
        .sum()
        .sort_values(by=column_x, ascending=ascending)
        .head(5)
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    
    colors = ["#90CAF9", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]

    sns.barplot(
        data=data,
        x=column_x,
        y=column_y,
        palette=colors,
        ax=ax
    )

    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Jumlah Terjual', fontsize=12)
    ax.set_ylabel('Kategori Produk', fontsize=12)
    
    st.pyplot(fig)
    plt.close(fig)

## Peta distribusi lokasi users
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


# VISUALISASI CHART ----------
## Buat DataFrame untuk analisis tren penjualan bulanan
monthly_sales_df = create_monthly_orders_df(filtered_df)

## Layout untuk menampilkan metrik
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_sales = monthly_sales_df['total_sales'].sum()
    st.metric(label="Total Sales", value=format_currency(total_sales, 'BRL', locale='pt_BR'))

with col2:
    avg_sales = monthly_sales_df['total_sales'].mean()
    st.metric(label="Avg. Sales per Month", value=format_currency(avg_sales, 'BRL', locale='pt_BR'))

with col3:
    total_orders = monthly_sales_df['total_orders'].sum()
    st.metric(label="Total Orders", value=total_orders)

with col4:
    avg_orders = monthly_sales_df['total_orders'].mean()
    st.metric(label="Avg. Orders per Month", value=round(avg_orders, 2))

## Tampilkan chart tren penjualan bulanan
st.subheader("📈 Tren Penjualan")
sales_trend_viz(monthly_sales_df['order_purchase_timestamp'], monthly_sales_df['total_sales'])

## Tampilkan chart produk terlaris dan terburuk
col1, col2 = st.columns(2)

with col1:
    plot_product_sales(
        data_df=filtered_df,
        column_y='product_category',
        column_x='quantity',
        ascending=False,
        title="Produk Terlaris"
    )

with col2:
    plot_product_sales(
        data_df=filtered_df,
        column_y='product_category',
        column_x='quantity',
        ascending=True,
        title="Produk Penjualan Terendah"
    )

## Tampilkan peta distribusi users
st.subheader("Distribusi Lokasi Users")
st_folium(create_users_map_df(customers_df, sellers_df), height=650)

st.caption("Copyright © 2026 - Data Science")