import streamlit as st
st.set_page_config(
    page_title="Dashboard", 
    layout="wide"
)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import geodatasets
import folium
import pydeck as pdk
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium
from babel.numbers import format_currency, get_currency_symbol
from matplotlib.ticker import FuncFormatter

sns.set_style("white")

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
        align-items: flex-end;
    }
    
    .logo-wrapper img {
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
## Formating angka metrik
def format_curr_short(value: float, currency: str = "BRL", locale:str = "pt_BR", decimals: int = 2) -> str:
    symbol = get_currency_symbol(currency, locale=locale)

    if value >= 1_000_000_000:
        short = f"{value / 1_000_000_000:.{decimals}f} B"
    elif value >= 1_000_000:
        short = f"{value / 1_000_000:.{decimals}f} M"
    elif value >= 1_000:
        short = f"{value / 1_000:.{decimals}f} K"
    else:
        short = f"{value:,.0f}"

    return f"{symbol}{short}"

## Fungsi untuk membuat DataFrame tren penjualan
def create_sales_trend_df(df, periode: str):
    df = df.copy()
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'], errors='coerce')

    sales_trend_df = (
        df.resample(rule=periode, on='order_purchase_timestamp')
        .agg({
            'order_id': 'nunique',
            'payment_value': 'sum'
        })
        .reset_index()
    )

    sales_trend_df.rename(columns={
        'order_id': 'total_orders',
        'payment_value': 'total_sales'
    }, inplace=True)

    if periode == 'W':
        sales_trend_df['order_purchase_timestamp'] = sales_trend_df['order_purchase_timestamp'].dt.strftime('W-%U %Y')
    elif periode == 'M':
        sales_trend_df['order_purchase_timestamp'] = sales_trend_df['order_purchase_timestamp'].dt.strftime('%b %y')
    elif periode == 'Q':
        sales_trend_df['order_purchase_timestamp'] = sales_trend_df['order_purchase_timestamp'].dt.to_period('Q').dt.strftime('Q%q %Y')
    elif periode == 'Y':
        sales_trend_df['order_purchase_timestamp'] = sales_trend_df['order_purchase_timestamp'].dt.strftime('%Y')
    else:
        raise ValueError("Periode tidak valid!")
    
    return sales_trend_df

## Formating angka y_axis tren penjualan
def y_axis_formatter(x, pos):
    if x >= 1_000_000_000:
        return f"{x / 1_000_000_000:.1f}B"
    elif x >= 1_000_000:
        return f"{x / 1_000_000:.1f}M"
    elif x >= 1_000:
        return f"{x / 1_000:.1f}K"
    else:
        return f"{x:,.0f}"

## Visualisasi tren penjualan
def sales_trend_viz(x, y, xlabel: str):
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(x, y, marker='o', linewidth=2, markersize=5, color="#6EC6BF")
    ax.set_xlabel(xlabel, fontweight='bold', color='white')
    ax.set_ylabel('Total Penjualan', fontweight='bold', color='white')

    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(visible=True, which='major', axis='y', color='gray', linestyle='--', alpha=0.7)

    # background transparan
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    st.pyplot(fig)
    plt.close(fig)

## Visualisasi penjualan produk
def plot_product_sales(data_df, ascending=False):
    data = (
        data_df
        .rename(columns={'product_category_name_english': 'product_category'})
        .groupby('product_category')
        .size()
        .reset_index(name='quantity')
        .sort_values(by='quantity', ascending=ascending)
        .head(5)
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    
    colors = ["#6EC6BF", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]

    sns.barplot(
        data=data,
        x='quantity',
        y='product_category',
        palette=colors,
        ax=ax
    )

    ax.set_xlabel('Jumlah Terjual', fontsize=12, fontweight='bold', color='white')
    ax.set_ylabel('Kategori Produk', fontsize=12, fontweight='bold', color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    plt.tight_layout()
    plt.grid(False)
    
    # background transparan
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    st.pyplot(fig)
    plt.close(fig)

## Peta distribusi lokasi users
@st.cache_data()
def plot_users_map(customers_df, sellers_df):
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_facecolor('#f5f5f5')  # light grey
    fig.patch.set_alpha(0) # transparan

    # Boundary
    url = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
    world = gpd.read_file(url)

    # Filter Brazil
    brazil = world[world["ADMIN"] == "Brazil"]

    brazil.plot(
        ax=plt.gca(), color='black', edgecolor='black', linewidth=0.5)

    ## Customer Layer
    hb = ax.hexbin(
        customers_df['geolocation_lng'], 
        customers_df['geolocation_lat'], 
        gridsize=80, 
        cmap='BuGn',
        mincnt=1, 
        alpha=0.8,
        bins='log'
    )
    cbar = fig.colorbar(hb, ax=ax)
    cbar.set_label('Jumlah Customers')

    ## Seller Layer
    ax.scatter(
        sellers_df['geolocation_lng'], 
        sellers_df['geolocation_lat'], 
        color='#FFA500', # warna orange
        s=8, 
        alpha=0.5, 
        label='Sellers'
    )
    
    # Map Style
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.set_xlabel('Longitude', fontsize=12, color='white')
    ax.set_ylabel('Latitude', fontsize=12, color='white')
    ax.legend(loc='upper right')

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.grid(False)
    plt.tight_layout()

    return fig

# VISUALISASI CHART ----------
## Buat DataFrame untuk analisis tren penjualan bulanan
with st.container():
    st.subheader("Ringkasan Transaksi", text_alignment="center")
    ## Layout untuk menampilkan metrik
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    with kpi1:
        with st.container(border=True, horizontal_alignment="center", vertical_alignment="center"):
            total_sales = filtered_df['payment_value'].sum()
            st.markdown(f"""
                <div style='text-align: center;'> 
                    <div style='font-size: 1rem;'>Total Sales</div>
                    <div style='font-size: 1.5rem; color: #6EC6BF;'>{format_curr_short(total_sales, currency='BRL', locale='pt_BR')}</div>
                </div>
            """, unsafe_allow_html=True)

    with kpi2:
        with st.container(border=True, horizontal_alignment="center", vertical_alignment="center"):
            avg_sales = filtered_df['payment_value'].mean()
            st.markdown(f"""
                <div style='text-align: center;'> 
                    <div style='font-size: 1rem;'>Avg. Sales</div>
                    <div style='font-size: 1.5rem; color: #6EC6BF;'>{format_curr_short(avg_sales, currency='BRL', locale='pt_BR')}</div>
                </div>
            """, unsafe_allow_html=True)

    with kpi3:
        with st.container(border=True, horizontal_alignment="center", vertical_alignment="top"):
            total_orders = filtered_df['order_id'].value_counts().sum()
            st.markdown(f"""
                <div style='text-align: center;'> 
                    <div style='font-size: 1rem;'>Total Orders</div>
                    <div style='font-size: 1.5rem; color: #6EC6BF;'>{total_orders}</div>
                </div>
            """, unsafe_allow_html=True)

    with kpi4:
        with st.container(border=True, horizontal_alignment="center", vertical_alignment="center"):
            num_customer = filtered_df['customer_id'].nunique()
            st.markdown(f"""
                <div style='text-align: center;'> 
                    <div style='font-size: 1rem;'>Total Customers</div>
                    <div style='font-size: 1.5rem; color: #6EC6BF;'>{num_customer}</div>
                </div>
            """, unsafe_allow_html=True)

with st.container():
    st.subheader("Kinerja Layanan", text_alignment="center")
    ## Layout untuk menampilkan metrik
    kpi5, kpi6, kpi7, kpi8 = st.columns(4)

    with kpi5:
        with st.container(border=True):
            popular_payment = filtered_df['payment_type'].value_counts().index[0]
            st.markdown(f"""
                <div style='text-align: center;'> 
                    <div style='font-size: 1rem;'>Popular Payment</div>
                    <div style='font-size: 1.5rem; color: #6EC6BF;'>{popular_payment}</div>
                </div>
            """, unsafe_allow_html=True)

    with kpi6:
        with st.container(border=True):
            unpopular_payment = filtered_df['payment_type'].value_counts().index[-1]
            st.markdown(f"""
                <div style='text-align: center;'> 
                    <div style='font-size: 1rem;'>Unpopular Payment</div>
                    <div style='font-size: 1.5rem; color: #6EC6BF;'>{unpopular_payment}</div>
                </div>
            """, unsafe_allow_html=True)

    with kpi7:
        with st.container(border=True):
            delivery_success_rate = (filtered_df['order_status'] == 'delivered').mean() * 100
            st.markdown(f"""
                <div style='text-align: center;'> 
                    <div style='font-size: 1rem;'>Delivery Success Rate</div>
                    <div style='font-size: 1.5rem; color: #6EC6BF;'>{delivery_success_rate:.2f}%</div>
                </div>
            """, unsafe_allow_html=True)

    with kpi8:
        with st.container(border=True):
            avg_review = filtered_df['review_score'].mean()
            st.markdown(f"""
                <div style='text-align: center;'> 
                    <div style='font-size: 1rem;'>Avg. Rating</div>
                    <div style='font-size: 1.5rem; color: #6EC6BF;'>{avg_review:.2f}</div>
                </div>
            """, unsafe_allow_html=True)

## Visualisasi Tren Penjualan
st.subheader("📈 Tren Penjualan")

tab1, tab2, tab3, tab4 = st.tabs(["Yearly", "Quarterly", "Monthly", "Weekly"])

### Tab 1: Tren Tahunan
with tab1:
    yearly_sales_df = create_sales_trend_df(filtered_df, periode='Y')
    sales_trend_viz(yearly_sales_df['order_purchase_timestamp'], yearly_sales_df['total_sales'], xlabel="Tahun")

### Tab 2: Tren Quarterly
with tab2:
    quarterly_sales_df = create_sales_trend_df(filtered_df, periode='Q')
    sales_trend_viz(quarterly_sales_df['order_purchase_timestamp'], quarterly_sales_df['total_sales'], xlabel="Quarter")

### Tab 3: Tren Bulanan
with tab3:
    monthly_sales_df = create_sales_trend_df(filtered_df, periode='M')
    sales_trend_viz(monthly_sales_df['order_purchase_timestamp'], monthly_sales_df['total_sales'], xlabel="Bulan")

### Tab 3: Tren Bulanan
with tab4:
    weekly_sales_df = create_sales_trend_df(filtered_df, periode='W')
    sales_trend_viz(weekly_sales_df['order_purchase_timestamp'], weekly_sales_df['total_sales'], xlabel="Minggu")

## Tampilkan chart produk terlaris dan terburuk
col1, col2 = st.columns(2)

with col1:
    with st.container():
        st.subheader("👍 Produk Terlaris")
        plot_product_sales(
            data_df=filtered_df,
            ascending=False
        )

with col2:
    with st.container():
        st.subheader("👎 Produk Kurang Laris")
        plot_product_sales(
            data_df=filtered_df,
            ascending=True
        )
'''
## Tampilkan peta distribusi users
with st.container(border=True):
    st.subheader("🌎 Distribusi Lokasi Users", text_alignment="center")

    fig = plot_users_map(customers_df, sellers_df)
    st.pyplot(fig)
'''
st.divider()
st.markdown(
    """
    <div style="text-align: center;">
        <p style="font-size: 0.8rem; color:grey;">
            Copyright © 2026 - Data Science
        </p>
    </div>
    """, unsafe_allow_html=True
)