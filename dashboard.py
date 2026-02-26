import streamlit as st
st.set_page_config(
    page_title="Dashboard", 
    layout="wide"
)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pydeck as pdk
from babel.numbers import format_currency, get_currency_symbol
from matplotlib.ticker import FuncFormatter

sns.set_style("white")

# PENGOLAHAN DATA ----------
## Dataset
@st.cache_data
def load_dataset(data_path):
    data_df = pd.read_csv(data_path)

    datetime_columns = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
        "shipping_limit_date"
    ]

    for col in datetime_columns:
        if col in data_df.columns:
            data_df[col] = pd.to_datetime(data_df[col], errors='coerce')
                
    if "order_purchase_timestamp" in data_df.columns:
        data_df = data_df.sort_values(by='order_purchase_timestamp')

    return data_df

## Load data
sales_data_df = load_dataset('sales_data.csv')
customers_df = load_dataset('customers_data.csv')
sellers_df = load_dataset('sellers_data.csv')

# DASHBOARD UI ----------
st.markdown(
    "<h1 style='text-align: center; font-size: 3.5rem;'>Dashboard E-Commerce OB</h1>", 
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

## Filter data yang akan digunakan
def filter_data(df):
    filtered_data_df = df[
        (df['order_purchase_timestamp'].dt.date >= start_date) & 
        (df['order_purchase_timestamp'].dt.date <= end_date)
    ].copy()

    return filtered_data_df

filtered_sales_df = filter_data(sales_data_df)

filtered_customers_df = filter_data(customers_df)

filtered_sellers_df = filter_data(sellers_df)


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
def axis_formatter(x, pos):
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

    ax.yaxis.set_major_formatter(FuncFormatter(axis_formatter))
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(visible=True, which='major', axis='y', color='gray', linestyle='--', alpha=0.7)

    # Background transparan
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

    ax.xaxis.set_major_formatter(FuncFormatter(axis_formatter))
    ax.set_xlabel('Jumlah Terjual', fontsize=12, fontweight='bold', color='white')
    ax.set_ylabel('Kategori Produk', fontsize=12, fontweight='bold', color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    plt.tight_layout()
    plt.grid(False)
    
    # Background transparan
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    st.pyplot(fig)
    plt.close(fig)

## Analisis RFM
@st.cache_data
def analyze_rfm(data_df):
    snapshot_date = data_df['order_purchase_timestamp'].max() + pd.Timedelta(days=1)

    rfm_df = data_df.groupby('customer_unique_id').agg({
        'order_purchase_timestamp': lambda x: (snapshot_date - x.max()).days,
        'order_id': 'nunique',
        'payment_value': 'sum'
    }).reset_index()

    rfm_df.columns = ['customer_unique_id', 'recency', 'frequency', 'monetary']

    # Binning recency
    rfm_df['cus_status'] = pd.cut(
        rfm_df['recency'],
        bins=[-1, 60, 90, 180, float('inf')],
        labels=['Active', 'Rarely Active', 'Need to Touch', 'Inactive']
    )

    # Binning frequency
    rfm_df['cus_activities'] = pd.cut(
        rfm_df['frequency'],
        bins=[-1, 0, 5, 10, float('inf')],
        labels=['Tidak Pernah', 'Jarang', 'Sering', 'Sangat Sering']
    )

    # Binning monetray
    rfm_df['cus_value'] = pd.cut(
        rfm_df['monetary'],
        bins=[-1, 100, 500, 1000, float('inf')],
        labels=['Low', 'Middle-Low', 'Middle-High', 'High']
    )
    
    return rfm_df

## Segmentasi Customer based on RFM data
@st.cache_data
def create_customer_segment(rfm_df):
    # Scoring
    rfm_df['cus_status_score'] = rfm_df['cus_status'].map({'Inactive': 1, 'Need to Touch': 2, 'Rarely Active': 3, 'Active': 4})
    rfm_df['cus_activities_score'] = rfm_df['cus_activities'].map({'Tidak Pernah': 1, 'Jarang': 2, 'Sering': 3, 'Sangat Sering': 4})
    rfm_df['cus_value_score'] = rfm_df['cus_value'].map({'Low': 1, 'Middle-Low': 2, 'Middle-High': 3, 'High': 4})

    # Ubah tipe data dari object (hasil binning) ke tipe integer
    rfm_df['cus_status_score'] = rfm_df['cus_status_score'].astype(int)
    rfm_df['cus_activities_score'] = rfm_df['cus_activities_score'].astype(int)
    rfm_df['cus_value_score'] = rfm_df['cus_value_score'].astype(int)

    # Total Score
    rfm_df['cus_rating'] = (
        rfm_df['cus_status_score'] * 0.2 +
        rfm_df['cus_activities_score'] * 0.3 +
        rfm_df['cus_value_score'] * 0.5
    )

    def customer_segment(row):
        if row['cus_rating'] > 3.3:
            return 'Super'
        elif row['cus_rating'] > 2.3:
            return 'Regular'
        elif row['cus_rating'] > 1.3:
            return 'Potential'
        else:
            return 'Risk'
    
    # Buat segmentasi
    rfm_df['segment'] = rfm_df.apply(customer_segment, axis=1)

    # Tambahkan kolom city dan state dari customers_df
    cus_seg_df = pd.merge(
        left=filtered_customers_df[['customer_unique_id', 'customer_city', 'customer_state']],
        right= rfm_df,
        on='customer_unique_id',
        how='left'
    )

    # Pastikan tidak ada null dan duplikat
    if cus_seg_df.isnull().sum().sum() > 0:
        cus_seg_df.dropna(inplace=True)
    
    if cus_seg_df.duplicated().sum() > 0:
        cus_seg_df.drop_duplicates(inplace=True)

    return cus_seg_df

## Visualisasi Distribusi Cluster
def plot_cluster_customers(data_df):
    data = (
        data_df
        .groupby('segment')
        .size()
        .reset_index(name='jumlah')
        .sort_values(by='jumlah', ascending=False)
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    
    segment_colors = {
        'Super': "#6EC6BF",
        'Regular': "#D3D3D3",
        'Potential': "#FFA500",
        'Risk': "#F50505"
    }

    sns.barplot(
        data=data,
        x='segment',
        y='jumlah',
        palette=segment_colors,
        ax=ax
    )

    ax.yaxis.set_major_formatter(FuncFormatter(axis_formatter))
    ax.set_xlabel('Segment', fontsize=12, fontweight='bold', color='white')
    ax.set_ylabel('Jumlah Customer', fontsize=12, fontweight='bold', color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    plt.tight_layout()
    plt.grid(False)
    
    # Background transparan
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    st.pyplot(fig)
    plt.close(fig)

## Visualisasi customer's top city
def plot_customer_top_city(data_df):
    data = (
        data_df
        .groupby(['customer_city', 'segment'])
        .size()
        .reset_index(name='jumlah')
        .sort_values(by='jumlah', ascending=False)
    )

    top_city = (
        data_df
        .groupby('customer_city')
        .size()
        .sort_values(ascending=False)
        .head(5).index
    )

    data = data[data['customer_city'].isin(top_city)]

    fig, ax = plt.subplots(figsize=(10, 4))
    
    segment_colors = {
        'Super': "#6EC6BF",
        'Regular': "#D3D3D3",
        'Potential': "#FFA500",
        'Risk': "#F50505"
    }

    sns.barplot(
        data=data,
        x='customer_city',
        y='jumlah',
        hue='segment',
        palette=segment_colors,
        ax=ax
    )

    ax.yaxis.set_major_formatter(FuncFormatter(axis_formatter))
    ax.set_xlabel('Kota', fontsize=12, fontweight='bold', color='white')
    ax.set_ylabel('Jumlah Customer', fontsize=12, fontweight='bold', color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    plt.xticks(rotation=10)
    plt.tight_layout()
    plt.grid(False)
    
    # Background transparan
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    st.pyplot(fig)
    plt.close(fig)

## Peta distribusi lokasi users
@st.cache_data
def plot_users_map(customers_df, sellers_df):
    # Customer data
    customers_map = (
        customers_df[['geolocation_lng', 'geolocation_lat']]
        .dropna()
        .copy()
    )

    # Seller data
    sellers_map = (
        sellers_df[['geolocation_lng', 'geolocation_lat']]
        .dropna()
        .copy()
    )

    n_points = len(customers_map)

    if n_points < 1_000:
        radius = 15_000
    elif n_points < 50_000:
        radius = 12_000
    else:
        radius = 10_000

    # Customer Layer
    customer_layer = pdk.Layer(
        "HexagonLayer",
        data=customers_map,
        get_position='[geolocation_lng, geolocation_lat]',
        radius=radius,
        extruded=False,
        pickable=True,
        coverage=0.8,
        colorRange= [
            [160, 224, 208, 200], # min semi transparan
            [125, 206, 196, 230],
            [110, 198, 191, 255] # max "#6EC6BF"
        ]
    )

    # Seller Layer
    seller_layer = pdk.Layer(
        "ScatterplotLayer",
        data=sellers_map,
        get_position='[geolocation_lng, geolocation_lat]',
        get_fill_color=[255, 165, 0],
        get_radius=3000,
        opacity=0.1,
        pickable=True,
    )

    # View State Brazil
    view_state = pdk.ViewState(
        latitude=customers_df['geolocation_lat'].mean(),
        longitude=customers_df['geolocation_lng'].mean(),
        zoom=5,
        pitch=0,
    )

    # Deck Object
    deck = pdk.Deck(
        layers=[customer_layer, seller_layer],
        initial_view_state=view_state,
        map_style="light"
    )

    return deck

# VISUALISASI DATA ----------

st.markdown("""
<style>
/* Center tab container */            
div[data-baseweb="tab-list"] {
    display: flex;
    justify-content: flex-start !important;
    gap: 0px;
}       

/* Default ukuran tab */
div[data-baseweb="tab-list"] button {
    flex: 1 1 0px;
    max-width: 100px;        
}            
            
/* Default tab style */          
button[data-baseweb="tab"] {
    font-size: 14px;
    font-weight: 700;
    padding: 1px 5px;
    border-radius: 5px;
    background-color: transparent;
    color: grey;
    border: 1px solid grey;
    transition: all 0.25s ease;
}

/* Active tab style */            
button[data-baseweb="tab"][aria-selected="true"] {
    background-color: transparent;
    border: 1px solid #FFA500;
    color: #FFA500 !important;
}                  

/* Hover effect */
button[data-baseweb="tab"]:hover {
    background-color: transparent;
    border: 1px solid #FFA500;
    color: #FFA500 !important;
}            
</style>
""", unsafe_allow_html=True)

sales_page, users_page = st.tabs(["Sales Page", "Users Page"])

st.markdown("""
<style>

.kpi-card {
    background: linear-gradient(#262730);
    padding: 0.5rem;
    border-radius: 5px;
    border: 1px solid white;
    box-shadow: 0 0 10px grey;
    transition: all 0.25s ease;
}
</style>
""", unsafe_allow_html=True)

# Halaman Sales
with sales_page:
    with st.container():
        st.subheader("Ringkasan Transaksi", text_alignment="center")
        ## Layout untuk menampilkan metrik
        kpi_sales_1, kpi_sales_2, kpi_sales_3, kpi_sales_4 = st.columns(4)

        with kpi_sales_1:
            with st.container(horizontal_alignment="center", vertical_alignment="center"):
                total_sales = filtered_sales_df['payment_value'].sum()
                st.markdown(f"""
                    <div class="kpi-card">
                        <div style='text-align: center;'> 
                            <div style='font-size: 1rem;'>Total Sales</div>
                            <div style='font-size: 2rem; color: #6EC6BF;'>
                                {format_curr_short(total_sales, currency='BRL', locale='pt_BR')}
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

        with kpi_sales_2:
            with st.container(horizontal_alignment="center", vertical_alignment="center"):
                avg_sales = filtered_sales_df['payment_value'].mean()
                st.markdown(f"""
                    <div class="kpi-card">    
                        <div style='text-align: center;'> 
                            <div style='font-size: 1rem;'>Avg. Sales</div>
                            <div style='font-size: 2rem; color: #6EC6BF;'>
                                {format_curr_short(avg_sales, currency='BRL', locale='pt_BR')}
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

        with kpi_sales_3:
            with st.container(horizontal_alignment="center", vertical_alignment="top"):
                total_orders = filtered_sales_df['order_id'].value_counts().sum()
                st.markdown(f"""
                    <div class="kpi-card">
                        <div style='text-align: center;'> 
                            <div style='font-size: 1rem;'>Total Orders</div>
                            <div style='font-size: 2rem; color: #6EC6BF;'>
                                {total_orders}
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

        with kpi_sales_4:
            with st.container(horizontal_alignment="center", vertical_alignment="center"):
                num_customer = filtered_sales_df['customer_id'].nunique()
                order_per_cus = total_orders / num_customer
                st.markdown(f"""
                    <div class="kpi-card">
                        <div style='text-align: center;'> 
                            <div style='font-size: 1rem;'>Avg. Order per Customer</div>
                            <div style='font-size: 2rem; color: #6EC6BF;'>
                                {order_per_cus:0.1f}
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

    with st.container():
        st.subheader("Kinerja Layanan", text_alignment="center")
        ## Layout untuk menampilkan metrik
        kpi_sales_5, kpi_sales_6, kpi_sales_7, kpi_sales_8 = st.columns(4)

        with kpi_sales_5:
            with st.container(horizontal_alignment="center", vertical_alignment="center"):
                delivery_success_rate = ((filtered_sales_df['order_status'] == 'delivered').sum() / len(filtered_sales_df['order_status']) * 100)
                st.markdown(f"""
                    <div class="kpi-card">
                        <div style='text-align: center;'> 
                            <div style='font-size: 1rem;'>Delivery Success Rate</div>
                            <div style='font-size: 2rem; color: #6EC6BF;'>
                                {delivery_success_rate:.2f}%
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

        with kpi_sales_6:
            with st.container(horizontal_alignment="center", vertical_alignment="center"):
                filtered_sales_df['days_to_delivered'] = (filtered_sales_df['order_delivered_customer_date'] - filtered_sales_df['order_purchase_timestamp']).dt.days
                avg_delivery_days = filtered_sales_df['days_to_delivered'].mean()
                st.markdown(f"""
                    <div class="kpi-card">
                        <div style='text-align: center;'> 
                            <div style='font-size: 1rem;'>Avg. Delivery Days</div>
                            <div style='font-size: 2rem; color: #6EC6BF;'>
                                {avg_delivery_days:.0f}
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

        with kpi_sales_7:
            with st.container(horizontal_alignment="center", vertical_alignment="center"):
                filtered_sales_df['estimated_delivery_days'] = (filtered_sales_df['order_estimated_delivery_date'] - filtered_sales_df['order_purchase_timestamp']).dt.days
                filtered_sales_df['delivery_performance'] = (filtered_sales_df['estimated_delivery_days'] - filtered_sales_df['days_to_delivered']).apply(lambda x: 'Early' if x > 0 else ('On-Time' if x == 0 else 'Late'))
                delivery_late_rate = ((filtered_sales_df['delivery_performance'] == 'Late').sum()) / len(filtered_sales_df['delivery_performance']) * 100

                st.markdown(f"""
                    <div class="kpi-card">
                        <div style='text-align: center;'> 
                            <div style='font-size: 1rem;'>Delivery Late Rate</div>
                            <div style='font-size: 2rem; color: #6EC6BF;'>
                                {delivery_late_rate:.2f}% 
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

        with kpi_sales_8:
            with st.container(horizontal_alignment="center", vertical_alignment="center"):
                avg_review = filtered_sales_df['review_score'].mean()
                st.markdown(f"""
                    <div class="kpi-card">
                        <div style='text-align: center;'> 
                            <div style='font-size: 1rem;'>Avg. Rating</div>
                            <div style='font-size: 2rem; color: #6EC6BF;'>
                                {avg_review:.2f}
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
    
    st.markdown(" ")

    ## Visualisasi Tren Penjualan
    st.subheader("📈 Tren Penjualan")

    tab1, tab2, tab3, tab4 = st.tabs(["Yearly", "Quarterly", "Monthly", "Weekly"])
    
    ### Tab 1: Tren Tahunan
    with tab1:
        yearly_sales_df = create_sales_trend_df(filtered_sales_df, periode='Y')
        sales_trend_viz(yearly_sales_df['order_purchase_timestamp'], yearly_sales_df['total_sales'], xlabel="Tahun")

    ### Tab 2: Tren Quarterly
    with tab2:
        quarterly_sales_df = create_sales_trend_df(filtered_sales_df, periode='Q')
        sales_trend_viz(quarterly_sales_df['order_purchase_timestamp'], quarterly_sales_df['total_sales'], xlabel="Quarter")

    ### Tab 3: Tren Bulanan
    with tab3:
        monthly_sales_df = create_sales_trend_df(filtered_sales_df, periode='M')
        sales_trend_viz(monthly_sales_df['order_purchase_timestamp'], monthly_sales_df['total_sales'], xlabel="Bulan")

    ### Tab 3: Tren Bulanan
    with tab4:
        weekly_sales_df = create_sales_trend_df(filtered_sales_df, periode='W')
        sales_trend_viz(weekly_sales_df['order_purchase_timestamp'], weekly_sales_df['total_sales'], xlabel="Minggu")
    st.markdown('</div>', unsafe_allow_html=True)

    ## Tampilkan chart produk terlaris dan terburuk
    col1, col2 = st.columns(2)

    with col1:
        with st.container():
            st.subheader("Produk Terlaris 👍")
            plot_product_sales(
                data_df=filtered_sales_df,
                ascending=False
            )

    with col2:
        with st.container():
            st.subheader("Produk Kurang Laris 👎")
            plot_product_sales(
                data_df=filtered_sales_df,
                ascending=True
            )
# Halaman Users: Customers & Sellers
with users_page:

    # Tampilkan KPI Users
    with st.container():
        st.subheader("Ringkasan Users", text_alignment="center")
        ## Layout untuk menampilkan metrik
        kpi_users_0_spc, kpi_users_1, kpi_users_2, kpi_users_3_spc= st.columns([1, 1, 1, 1])

        with kpi_users_1:
            with st.container(horizontal_alignment="center", vertical_alignment="center"):
                total_customers = (filtered_customers_df['customer_unique_id'].nunique())
                st.markdown(f"""
                    <div class="kpi-card">
                        <div style='text-align: center;'> 
                            <div style='font-size: 1rem;'>Total Customers</div>
                            <div style='font-size: 2rem; color: #6EC6BF;'>
                                {total_customers}
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

        with kpi_users_2:
            with st.container(horizontal_alignment="center", vertical_alignment="center"):
                total_sellers = (filtered_sellers_df['seller_id'].nunique())
                st.markdown(f"""
                    <div class="kpi-card">
                        <div style='text-align: center;'> 
                            <div style='font-size: 1rem;'>Total Sellers</div>
                            <div style='font-size: 2rem; color: #6EC6BF;'>
                                {total_sellers}
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

    ## Tampilkan chart RFM dan Clustering
    col1, col2 = st.columns(2)
    
    ### Hitung RFM
    rfm_df = analyze_rfm(filtered_customers_df)
    ### Clustering
    cus_seg_df = create_customer_segment(rfm_df)

    with col1:
        with st.container():
            st.subheader("Profil Customer")
            plot_cluster_customers(cus_seg_df)

    with col2:
        with st.container():
            st.subheader("Top Kota by Customers")
            plot_customer_top_city(cus_seg_df)

    ## Tampilkan peta persebaran lokasi users
    with st.container(border=True):
        st.subheader("🌎 Persebaran Lokasi Users", text_alignment="center")

        deck = plot_users_map(filtered_customers_df, filtered_sellers_df)
        st.pydeck_chart(deck)

        st.markdown("""
            <div style="display:flex; justify-content:center; gap:20px; margin-top:5px; margin-bottom:5px;">
                <div style="display:flex; align-items:center; gap:5px;">
                    <div style="width:20px; height:20px; background-color:#6EC6BF;"></div>
                    <span>Customer</span>
                </div>
                <div style="display:flex; align-items:center; gap:5px;">
                    <div style="width:20px; height:20px; background-color:#FF6B00;"></div>
                    <span>Seller</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

with st.container():
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