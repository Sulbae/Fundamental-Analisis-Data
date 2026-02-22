import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import folium
from streamlit_folium import st_folium

from babel.numbers import format_currency
from datetime import datetime

sns.set_style("darkgrid")

def create_monthly_orders_df(df):
    monthly_orders_df = df.resample(rule='M', on='order_purchase_timestamp').agg({
        'order_id': 'nunique',
        'sales': 'sum'
    }).reset_index()
    
    monthly_orders_df.rename(columns={
        'order_id': 'total_orders',
        'sales': 'total_sales'
    }, inplace=True)

    monthly_orders_df['order_purchase_timestamp'] = monthly_orders_df['order_purchase_timestamp'].dt.strftime('%m-%Y')
    
    return monthly_orders_df

def create_product_sales_df(df):
    product_sales_df = df.groupby('product_category_name_english')['quantity'].sum().reset_index()
    product_sales_df.rename(columns={'product_category_name_english': 'product_category', 'quantity': 'total_quantity'}, inplace=True)
    product_sales_df.sort_values(by='total_quantity', ascending=False, inplace=True)
    
    return product_sales_df

sales_data_df = pd.read_csv('sales_data.csv')

datetime_columns = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
    "shipping_limit_date"
]

sales_data_df.sort_values(by='order_purchase_timestamp', inplace=True).reset_index(inplace=True)

for col in datetime_columns:
    sales_data_df[col] = pd.to_datetime(sales_data_df[col], errors='coerce')

# Komponen filter waktu
min_date = sales_data_df['order_purchase_timestamp'].min().date()
max_date = sales_data_df['order_purchase_timestamp'].max().date()

st.sidebar.header("Periode Penjualan")
start_date = st.sidebar.date_input("Start Date", min_value=min_date, max_value=max_date, value=min_date)
end_date = st.sidebar.date_input("End Date", min_value=min_date, max_value=max_date, value=max_date)

# Simpan data terfilter yang akan digunakan
filtered_df = sales_data_df[(sales_data_df['order_purchase_timestamp'].dt.date >= start_date) & 
                          (sales_data_df['order_purchase_timestamp'].dt.date <= end_date)].copy()


# Buat DataFrame untuk analisis tren penjualan bulanan
monthly_sales_df = create_monthly_orders_df(filtered_df)

# Buat DataFrame untuk analisis produk terlaris
product_sales_df = create_product_sales_df(filtered_df)


# Dashboard Streamlit
st.title("Dashboard Penjualan E-Commerce")

st.subheader("Tren Penjualan")

col1, col2 = st.columns(2)

with col1:
    total_sales = monthly_sales_df['total_sales'].sum()
    st.metric(label="Total Sales", value=format_currency(total_sales, 'BRL', locale='pt_BR'))

with col2:
    total_orders = monthly_sales_df['total_orders'].sum()
    st.metric(label="Total Orders", value=total_orders)

# Visualisasi tren penjualan bulanan
def sales_trend_viz(x, y, title):
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, marker='o', linewidth=2, markersize=8, color="#90CAF9")
    plt.title(title)
    plt.xlabel('Bulan')
    plt.ylabel('Total Penjualan')
    plt.xticks(rotation=45)
    
    st.pyplot(plt)

sales_trend_viz(monthly_sales_df['order_purchase_timestamp'], monthly_sales_df['total_sales'], title='Tren Penjualan Bulanan')

# Visualisasi produk terlaris
def top_product_viz(data_df, column_x, column_y):
    plt.figure(figsize=(10, 5))
    
    colors = ["#90CAF9", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]

    sns.barplot(
        x=data_df[column_x],
        y=data_df[column_y],
        data=data_df.head(5),
        palette=colors
    )

    plt.title('Top 5 Produk Terlaris', fontsize=16, loc="center")
    plt.xlabel('Jumlah Terjual', fontsize=12)
    plt.ylabel('Kategori Produk', fontsize=12)
    
    st.pyplot(plt)

top_product_viz(product_sales_df, column_x='total_quantity', column_y='product_category')


st.caption("Copyright © 2026.")