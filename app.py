import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="E-Commerce Sales Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&family=DM+Mono:wght@400;500&display=swap');

/* Root theme */
:root {
    --navy: #0a0f1e;
    --navy-light: #111827;
    --navy-card: #151f35;
    --cyan: #00d4ff;
    --cyan-dim: rgba(0, 212, 255, 0.15);
    --cyan-border: rgba(0, 212, 255, 0.3);
    --white: #f0f4ff;
    --muted: #8899aa;
    --green: #00e5a0;
    --orange: #ff7b54;
    --font-display: 'Syne', sans-serif;
    --font-body: 'DM Sans', sans-serif;
    --font-mono: 'DM Mono', monospace;
}

/* Global overrides */
html, body, [class*="css"] {
    font-family: var(--font-body) !important;
    background-color: var(--navy) !important;
    color: var(--white) !important;
}

.stApp {
    background: linear-gradient(135deg, #0a0f1e 0%, #0d1526 50%, #0a0f1e 100%) !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--navy-card) !important;
    border-right: 1px solid var(--cyan-border) !important;
}
[data-testid="stSidebar"] * {
    color: var(--white) !important;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: var(--navy-card) !important;
    border: 1px solid var(--cyan-border) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}
[data-testid="stMetricValue"] {
    font-family: var(--font-display) !important;
    font-weight: 700 !important;
    color: var(--cyan) !important;
    font-size: 1.8rem !important;
}
[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
    font-size: 0.8rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}

/* Headers */
h1, h2, h3 {
    font-family: var(--font-display) !important;
    color: var(--white) !important;
}

/* DataFrames */
[data-testid="stDataFrame"] {
    border: 1px solid var(--cyan-border) !important;
    border-radius: 8px !important;
}

/* Tabs */
[data-testid="stTab"] {
    font-family: var(--font-display) !important;
    font-weight: 600 !important;
}

/* Buttons */
.stButton > button {
    background: var(--cyan-dim) !important;
    border: 1px solid var(--cyan-border) !important;
    color: var(--cyan) !important;
    font-family: var(--font-display) !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: var(--cyan) !important;
    color: var(--navy) !important;
}

/* Info/Success boxes */
.stAlert {
    background: var(--navy-card) !important;
    border-radius: 10px !important;
}

/* Select/slider widgets */
.stSelectbox > div > div,
.stSlider {
    background: var(--navy-card) !important;
}

/* Dividers */
hr {
    border-color: var(--cyan-border) !important;
}

/* Section Cards */
.section-card {
    background: var(--navy-card);
    border: 1px solid var(--cyan-border);
    border-radius: 14px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
}

.hero-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 3rem;
    background: linear-gradient(90deg, #00d4ff, #f0f4ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
    margin-bottom: 0.25rem;
}

.hero-subtitle {
    font-family: 'DM Sans', sans-serif;
    font-weight: 300;
    font-size: 1.1rem;
    color: #8899aa;
    margin-bottom: 0;
}

.badge {
    display: inline-block;
    background: rgba(0, 212, 255, 0.12);
    border: 1px solid rgba(0, 212, 255, 0.3);
    color: #00d4ff;
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    margin-right: 0.5rem;
    margin-bottom: 0.5rem;
}

.insight-card {
    background: rgba(0, 229, 160, 0.07);
    border: 1px solid rgba(0, 229, 160, 0.25);
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
}

.insight-card strong {
    color: #00e5a0;
}

.warning-card {
    background: rgba(255, 123, 84, 0.07);
    border: 1px solid rgba(255, 123, 84, 0.25);
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
}

.model-winner {
    background: linear-gradient(135deg, rgba(0, 212, 255, 0.1), rgba(0, 229, 160, 0.1));
    border: 1px solid rgba(0, 212, 255, 0.4);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    text-align: center;
}

.step-badge {
    background: var(--cyan);
    color: var(--navy);
    font-family: 'DM Mono', monospace;
    font-weight: 700;
    font-size: 0.8rem;
    width: 1.8rem;
    height: 1.8rem;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    margin-right: 0.6rem;
}

.code-block {
    background: #0d1526;
    border: 1px solid rgba(0, 212, 255, 0.2);
    border-left: 3px solid #00d4ff;
    border-radius: 6px;
    padding: 0.8rem 1rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.82rem;
    color: #a8d8ff;
    margin: 0.5rem 0;
    overflow-x: auto;
}
</style>
""", unsafe_allow_html=True)

# ─── Plotly theme ────────────────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", color="#f0f4ff"),
    xaxis=dict(gridcolor="rgba(0,212,255,0.08)", linecolor="rgba(0,212,255,0.2)", tickfont_color="#8899aa"),
    yaxis=dict(gridcolor="rgba(0,212,255,0.08)", linecolor="rgba(0,212,255,0.2)", tickfont_color="#8899aa"),
    margin=dict(l=10, r=10, t=50, b=10),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,212,255,0.2)", borderwidth=1),
)

COLORS = {
    "cyan": "#00d4ff",
    "green": "#00e5a0",
    "orange": "#ff7b54",
    "purple": "#c084fc",
    "yellow": "#fcd34d",
}

# ─── Data Loading & Processing ───────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_process():
    orders = pd.read_csv("datasets/olist_orders_dataset.csv")
    order_items = pd.read_csv("datasets/olist_order_items_dataset.csv")

    df_orders = orders[["order_id", "order_purchase_timestamp"]].copy()
    df_order_items = order_items[["order_id", "price", "freight_value"]].copy()

    df_orders["order_purchase_timestamp"] = pd.to_datetime(
        df_orders["order_purchase_timestamp"], errors="coerce"
    )
    df_orders["date"] = df_orders["order_purchase_timestamp"].dt.date

    df = pd.merge(df_orders, df_order_items, on="order_id", how="inner")
    df["revenue"] = df["price"] + df["freight_value"]

    daily_sales = df.groupby("date")["revenue"].sum().reset_index()
    daily_sales["date"] = pd.to_datetime(daily_sales["date"], errors="coerce")
    daily_sales["month"] = daily_sales["date"].dt.month
    daily_sales["day"] = daily_sales["date"].dt.day
    daily_sales["weekday"] = daily_sales["date"].dt.weekday
    daily_sales["is_weekend"] = daily_sales["weekday"].isin([5, 6]).astype(int)

    train = daily_sales[daily_sales["date"] < "2018-01-01"].copy()
    test = daily_sales[daily_sales["date"] >= "2018-01-01"].copy()

    features = ["month", "day", "weekday", "is_weekend"]
    x_train = train[features]
    y_train = train["revenue"]
    x_test = test[features]
    y_test = test["revenue"]

    lr = LinearRegression()
    lr.fit(x_train, y_train)
    lr_preds = lr.predict(x_test)

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(x_train, y_train)
    rf_preds = rf.predict(x_test)

    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))

    return (
        orders, order_items, df, daily_sales,
        train, test,
        y_test, lr_preds, rf_preds,
        lr_rmse, rf_rmse,
        lr, rf, features
    )

# ─── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 1.5rem'>
        <div style='font-family: Syne; font-size: 1.4rem; font-weight: 800; color: #00d4ff;'>📈 Sales Forecast</div>
        <div style='font-size: 0.75rem; color: #8899aa; font-family: DM Mono; margin-top: 0.3rem;'>Olist · Brazilian E-Commerce</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("<div style='font-family: Syne; font-weight: 700; font-size: 0.9rem; color: #8899aa; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.5rem;'>Navigation</div>", unsafe_allow_html=True)
    
    page = st.radio(
        "",
        ["🏠  Overview", "🔍  EDA", "⚙️  Feature Engineering", "🤖  Model Training", "📊  Results & Evaluation", "💡  Business Insights"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("""
    <div style='font-family: DM Sans; font-size: 0.82rem; color: #8899aa; padding: 0.5rem 0;'>
        <div style='margin-bottom: 0.6rem;'><span class='badge'>Dataset</span> Olist Brazil</div>
        <div style='margin-bottom: 0.6rem;'><span class='badge'>Orders</span> 99,441 rows</div>
        <div style='margin-bottom: 0.6rem;'><span class='badge'>Items</span> 112,650 rows</div>
        <div style='margin-bottom: 0.6rem;'><span class='badge'>Target</span> Daily Revenue</div>
        <div><span class='badge'>Type</span> Regression</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <div style='font-size: 0.75rem; color: #556677; text-align: center; padding-top: 0.5rem; font-family: DM Mono;'>
        github.com/shaikhsanan04<br>ecommerce-sales-forecasting
    </div>
    """, unsafe_allow_html=True)

# ─── Load Data ───────────────────────────────────────────────────────────────────
with st.spinner("Loading & processing data..."):
    try:
        (orders, order_items, df, daily_sales,
         train, test,
         y_test, lr_preds, rf_preds,
         lr_rmse, rf_rmse,
         lr_model, rf_model, features) = load_and_process()
        data_loaded = True
    except Exception as e:
        data_loaded = False
        load_error = str(e)

# ─── Helper: section header ───────────────────────────────────────────────────────
def section_header(icon, title, subtitle=""):
    sub_html = f"<div style='font-family: DM Sans; font-size: 1rem; color: #8899aa; margin-top: 0.3rem; font-weight: 300;'>{subtitle}</div>" if subtitle else ""
    st.markdown(f"""
    <div style='margin-bottom: 2rem; padding-bottom: 1rem; border-bottom: 1px solid rgba(0,212,255,0.15);'>
        <div style='font-family: Syne; font-size: 2rem; font-weight: 800; color: #f0f4ff;'>{icon} {title}</div>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE: Overview
# ════════════════════════════════════════════════════════════════════════════════
if "Overview" in page:
    st.markdown("""
    <div style='padding: 2.5rem 0 1.5rem;'>
        <div class='hero-title'>E-Commerce Sales<br>Forecasting</div>
        <div class='hero-subtitle'>Predicting daily revenue using the Olist Brazilian E-Commerce dataset</div>
        <div style='margin-top: 1.2rem;'>
            <span class='badge'>Python</span>
            <span class='badge'>Scikit-Learn</span>
            <span class='badge'>Pandas</span>
            <span class='badge'>Plotly</span>
            <span class='badge'>Streamlit</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not data_loaded:
        st.error(f"Could not load datasets. Make sure `datasets/` folder is in the project root.\n\n`{load_error}`")
        st.stop()

    # KPI row
    total_rev = daily_sales["revenue"].sum()
    avg_daily = daily_sales["revenue"].mean()
    total_days = len(daily_sales)
    best_day_rev = daily_sales["revenue"].max()
    best_day_date = daily_sales.loc[daily_sales["revenue"].idxmax(), "date"]
    winner = "Linear Regression" if lr_rmse < rf_rmse else "Random Forest"
    winner_rmse = min(lr_rmse, rf_rmse)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Revenue", f"R$ {total_rev:,.0f}", "Entire dataset period")
    c2.metric("Avg Daily Revenue", f"R$ {avg_daily:,.0f}", f"{total_days} days tracked")
    c3.metric("Best Single Day", f"R$ {best_day_rev:,.0f}", str(best_day_date.date()))
    c4.metric("Best Model RMSE", f"R$ {winner_rmse:,.0f}", winner)

    st.markdown("<br>", unsafe_allow_html=True)

    # Quick revenue chart
    fig = px.area(
        daily_sales, x="date", y="revenue",
        title="Daily Revenue — Full Timeline",
        labels={"revenue": "Revenue (BRL)", "date": ""},
        color_discrete_sequence=[COLORS["cyan"]],
    )
    fig.update_traces(
        line=dict(width=1.5),
        fillcolor="rgba(0,212,255,0.07)",
    )
    fig.update_layout(**PLOT_LAYOUT, title_font=dict(family="Syne", size=18, color="#f0f4ff"))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Project description
    col1, col2 = st.columns([1.1, 1])
    with col1:
        st.markdown("""
        <div class='section-card'>
            <div style='font-family: Syne; font-weight: 700; font-size: 1.15rem; color: #00d4ff; margin-bottom: 1rem;'>🎯 Project Objective</div>
            <p style='color: #c8d8e8; line-height: 1.7;'>
                Build a machine learning pipeline that <strong style='color:#f0f4ff;'>predicts daily e-commerce sales revenue</strong>
                from historical transaction data. The model learns calendar-driven patterns and helps
                businesses plan ahead with data-driven confidence.
            </p>
            <div style='font-family: Syne; font-weight: 700; font-size: 1rem; color: #00d4ff; margin: 1.2rem 0 0.6rem;'>📦 Dataset</div>
            <p style='color: #c8d8e8; line-height: 1.7;'>
                The <strong style='color:#f0f4ff;'>Olist Brazilian E-Commerce Public Dataset</strong> contains real transaction
                records from 2016–2018. Two tables are joined:
            </p>
            <ul style='color: #c8d8e8; line-height: 1.9;'>
                <li><code style='color:#00d4ff;'>olist_orders_dataset.csv</code> — order timestamps</li>
                <li><code style='color:#00d4ff;'>olist_order_items_dataset.csv</code> — price &amp; freight</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='section-card'>
            <div style='font-family: Syne; font-weight: 700; font-size: 1.15rem; color: #00d4ff; margin-bottom: 1rem;'>🔄 Project Workflow</div>
        """, unsafe_allow_html=True)
        steps = [
            "Load & inspect raw datasets",
            "Select relevant columns",
            "Clean and convert datatypes",
            "Merge on order_id",
            "Engineer revenue feature",
            "Aggregate to daily granularity",
            "Extract time-based features",
            "Exploratory data analysis",
            "Chronological train/test split",
            "Train Linear Regression & Random Forest",
            "Evaluate with RMSE",
            "Visualize predictions & insights",
        ]
        for i, s in enumerate(steps, 1):
            st.markdown(f"""
            <div style='display:flex; align-items:center; margin-bottom:0.45rem;'>
                <span style='background:#00d4ff;color:#0a0f1e;font-family:DM Mono;font-weight:700;font-size:0.7rem;
                    width:1.4rem;height:1.4rem;border-radius:50%;display:inline-flex;align-items:center;
                    justify-content:center;flex-shrink:0;margin-right:0.6rem;'>{i}</span>
                <span style='color:#c8d8e8;font-size:0.9rem;'>{s}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE: EDA
# ════════════════════════════════════════════════════════════════════════════════
elif "EDA" in page:
    if not data_loaded:
        st.error(f"Could not load datasets.\n\n`{load_error}`")
        st.stop()

    section_header("🔍", "Exploratory Data Analysis", "Understanding patterns, distributions, and trends in daily revenue")

    # Raw data preview
    st.markdown("### 📋 Raw Data Preview")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div style='font-family:DM Mono;font-size:0.8rem;color:#00d4ff;margin-bottom:0.4rem;'>olist_orders_dataset.csv — 99,441 rows × 8 cols</div>", unsafe_allow_html=True)
        st.dataframe(orders.head(5), use_container_width=True, height=210)
    with col2:
        st.markdown("<div style='font-family:DM Mono;font-size:0.8rem;color:#00d4ff;margin-bottom:0.4rem;'>olist_order_items_dataset.csv — 112,650 rows × 7 cols</div>", unsafe_allow_html=True)
        st.dataframe(order_items.head(5), use_container_width=True, height=210)

    st.markdown("<br>", unsafe_allow_html=True)

    # Data quality
    st.markdown("### 🧹 Data Quality Check")
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Orders Missing Values", "160", "order_approved_at only")
    q2.metric("Items Missing Values", "0", "Clean ✓")
    q3.metric("Order Duplicates", "0", "Unique orders")
    q4.metric("Item Duplicates", "13,984", "Expected (multi-item orders)")

    st.info("ℹ️  Item-level duplicates on `order_id` are **expected** — one order can contain multiple products. These are not data errors.", icon=None)

    st.markdown("<br>", unsafe_allow_html=True)

    # Descriptive stats
    st.markdown("### 📐 Descriptive Statistics — Merged Dataset")
    stat_df = df[["price", "freight_value", "revenue"]].describe().round(2)
    stat_df.columns = ["Price (BRL)", "Freight Value (BRL)", "Revenue per Item (BRL)"]
    st.dataframe(stat_df, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Revenue distributions
    st.markdown("### 📊 Revenue Distribution")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(
            df[df["revenue"] < 800], x="revenue", nbins=80,
            title="Per-Item Revenue Distribution",
            labels={"revenue": "Revenue per Item (BRL)"},
            color_discrete_sequence=[COLORS["cyan"]],
        )
        fig.update_layout(**PLOT_LAYOUT, title_font=dict(family="Syne", size=15, color="#f0f4ff"))
        fig.update_traces(marker_line_color="rgba(0,212,255,0.3)", marker_line_width=0.5)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig2 = px.histogram(
            daily_sales, x="revenue", nbins=60,
            title="Daily Aggregated Revenue Distribution",
            labels={"revenue": "Daily Revenue (BRL)"},
            color_discrete_sequence=[COLORS["green"]],
        )
        fig2.update_layout(**PLOT_LAYOUT, title_font=dict(family="Syne", size=15, color="#f0f4ff"))
        fig2.update_traces(marker_line_color="rgba(0,229,160,0.3)", marker_line_width=0.5)
        st.plotly_chart(fig2, use_container_width=True)

    # Daily trend
    st.markdown("### 📈 Daily Revenue Trend")
    fig3 = px.area(
        daily_sales, x="date", y="revenue",
        title="Daily Sales Revenue Over Time (2016–2018)",
        labels={"revenue": "Revenue (BRL)", "date": ""},
        color_discrete_sequence=[COLORS["cyan"]],
    )
    fig3.update_traces(line=dict(width=1.5), fillcolor="rgba(0,212,255,0.06)")
    # Add train/test split line
    fig3.add_vline(
        x="2018-01-01", line_dash="dash", line_color="#fcd34d",
        annotation_text="Train / Test Split (Jan 2018)",
        annotation_font_color="#fcd34d",
        annotation_font_family="DM Mono",
    )
    fig3.update_layout(**PLOT_LAYOUT, title_font=dict(family="Syne", size=16, color="#f0f4ff"))
    st.plotly_chart(fig3, use_container_width=True)

    # Monthly & weekday
    st.markdown("### 🗓️ Temporal Patterns")
    col1, col2 = st.columns(2)

    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                   7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    monthly_sales = daily_sales.groupby("month")["revenue"].mean().reset_index()
    monthly_sales["month_name"] = monthly_sales["month"].map(month_names)

    with col1:
        fig4 = px.bar(
            monthly_sales, x="month_name", y="revenue",
            title="Average Daily Revenue by Month",
            labels={"revenue": "Avg Revenue (BRL)", "month_name": "Month"},
            color="revenue",
            color_continuous_scale=[[0,"#0a2040"],[0.5,"#00d4ff"],[1,"#00e5a0"]],
        )
        fig4.update_layout(
            **PLOT_LAYOUT,
            title_font=dict(family="Syne", size=15, color="#f0f4ff"),
            coloraxis_showscale=False,
            xaxis_categoryorder="array",
            xaxis_categoryarray=list(month_names.values()),
        )
        st.plotly_chart(fig4, use_container_width=True)

    day_names = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}
    weekday_sales = daily_sales.groupby("weekday")["revenue"].mean().reset_index()
    weekday_sales["day_name"] = weekday_sales["weekday"].map(day_names)

    with col2:
        weekday_sales["is_weekend"] = weekday_sales["weekday"].isin([5, 6])
        fig5 = px.bar(
            weekday_sales, x="day_name", y="revenue",
            title="Average Daily Revenue by Weekday",
            labels={"revenue": "Avg Revenue (BRL)", "day_name": "Day"},
            color="is_weekend",
            color_discrete_map={False: COLORS["cyan"], True: COLORS["orange"]},
        )
        fig5.update_layout(
            **PLOT_LAYOUT,
            title_font=dict(family="Syne", size=15, color="#f0f4ff"),
            showlegend=False,
        )
        st.plotly_chart(fig5, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE: Feature Engineering
# ════════════════════════════════════════════════════════════════════════════════
elif "Feature Engineering" in page:
    if not data_loaded:
        st.error(f"Could not load datasets.\n\n`{load_error}`")
        st.stop()

    section_header("⚙️", "Feature Engineering", "Transforming raw timestamps into model-ready inputs")

    st.markdown("""
    <div class='section-card'>
        <div style='font-family: Syne; font-weight: 700; font-size: 1.1rem; color: #00d4ff; margin-bottom: 0.8rem;'>Why Feature Engineering?</div>
        <p style='color: #c8d8e8; line-height: 1.7;'>
            Raw timestamps cannot be fed directly into regression models. We extract meaningful
            numeric signals that capture <strong style='color:#f0f4ff;'>cyclical and seasonal patterns</strong> in
            e-commerce buying behavior. These hand-crafted features give the model the temporal
            context it needs to learn revenue patterns.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Pipeline steps
    st.markdown("### 🔧 Engineering Pipeline")

    steps_fe = [
        ("Revenue per Item", "price + freight_value", "Combines product cost and shipping into a single revenue signal per item."),
        ("Daily Aggregation", "groupby(date)[revenue].sum()", "Collapses transaction-level data into one row per day."),
        ("Month", "date.dt.month → int 1–12", "Captures monthly seasonality patterns."),
        ("Day of Month", "date.dt.day → int 1–31", "Captures within-month patterns (e.g. pay-day effects)."),
        ("Weekday", "date.dt.weekday → int 0–6", "Monday=0, Sunday=6. Captures weekly purchase rhythms."),
        ("Is Weekend", "weekday.isin([5,6]).astype(int)", "Binary flag. Weekends often see different buying behavior."),
        ("Train/Test Split", "date < '2018-01-01'", "Chronological split — never shuffle time series data."),
    ]

    for name, code, desc in steps_fe:
        st.markdown(f"""
        <div style='display:flex; gap:1rem; margin-bottom:1rem; align-items:flex-start;'>
            <div style='background:rgba(0,212,255,0.12);border:1px solid rgba(0,212,255,0.3);border-radius:8px;
                padding:0.6rem 1rem;min-width:170px;text-align:center;'>
                <div style='font-family:Syne;font-weight:700;font-size:0.85rem;color:#00d4ff;'>{name}</div>
            </div>
            <div style='flex:1;'>
                <div class='code-block'>{code}</div>
                <div style='color:#8899aa;font-size:0.85rem;margin-top:0.25rem;'>{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Resulting feature set
    st.markdown("### 📋 Resulting Feature Table (Sample)")
    sample = daily_sales[["date", "month", "day", "weekday", "is_weekend", "revenue"]].head(10).copy()
    sample["date"] = sample["date"].dt.strftime("%Y-%m-%d")
    sample.columns = ["Date", "Month", "Day", "Weekday (0=Mon)", "Is Weekend", "Daily Revenue (BRL)"]
    st.dataframe(sample, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Train / Test split viz
    st.markdown("### ✂️ Train / Test Split")
    train_size = len(train)
    test_size = len(test)

    col1, col2, col3 = st.columns(3)
    col1.metric("Training Days", train_size, "Pre-2018 data")
    col2.metric("Test Days", test_size, "2018 onwards")
    col3.metric("Train Ratio", f"{train_size/(train_size+test_size)*100:.0f}%", "Chronological split")

    st.markdown(f"""
    <div style='margin: 1rem 0;'>
        <div style='display:flex; height:12px; border-radius:6px; overflow:hidden; background:#151f35;'>
            <div style='width:{train_size/(train_size+test_size)*100:.0f}%;background:linear-gradient(90deg,#00d4ff,#00a8cc);'></div>
            <div style='flex:1;background:linear-gradient(90deg,#ff7b54,#ff5533);'></div>
        </div>
        <div style='display:flex;justify-content:space-between;margin-top:0.4rem;font-family:DM Mono;font-size:0.75rem;color:#8899aa;'>
            <span>🟦 Train ({train_size} days)</span>
            <span>🟧 Test ({test_size} days)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='insight-card'>
        <strong>Why chronological split?</strong> In time series forecasting, randomly shuffling and splitting
        would create <em>data leakage</em> — the model would see future data during training, inflating its
        apparent performance. We always split along the time axis.
    </div>
    """, unsafe_allow_html=True)

    # Correlation
    st.markdown("### 🔗 Feature Correlation with Revenue")
    corr = daily_sales[["month", "day", "weekday", "is_weekend", "revenue"]].corr()["revenue"].drop("revenue")
    corr_df = corr.reset_index()
    corr_df.columns = ["Feature", "Correlation"]
    corr_df["Color"] = corr_df["Correlation"].apply(lambda x: COLORS["cyan"] if x > 0 else COLORS["orange"])

    fig_corr = go.Figure(go.Bar(
        x=corr_df["Correlation"], y=corr_df["Feature"],
        orientation="h",
        marker_color=corr_df["Color"],
        text=corr_df["Correlation"].round(3),
        textposition="outside",
        textfont=dict(family="DM Mono", color="#f0f4ff", size=11),
    ))
    fig_corr.update_layout(
        **PLOT_LAYOUT,
        title="Pearson Correlation of Features vs Daily Revenue",
        title_font=dict(family="Syne", size=16, color="#f0f4ff"),
        yaxis=dict(tickfont=dict(family="DM Mono", color="#c8d8e8")),
        height=300,
    )
    st.plotly_chart(fig_corr, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE: Model Training
# ════════════════════════════════════════════════════════════════════════════════
elif "Model Training" in page:
    if not data_loaded:
        st.error(f"Could not load datasets.\n\n`{load_error}`")
        st.stop()

    section_header("🤖", "Model Training", "Linear Regression vs Random Forest Regressor")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class='section-card'>
            <div style='font-family:Syne;font-weight:800;font-size:1.2rem;color:#00d4ff;margin-bottom:0.8rem;'>
                📐 Linear Regression
            </div>
            <p style='color:#c8d8e8;line-height:1.7;font-size:0.9rem;'>
                Assumes a <strong style='color:#f0f4ff;'>linear relationship</strong> between input features and the
                target variable (revenue). Fits a plane through the feature space to minimize squared errors.
            </p>
            <div style='margin-top:1rem;'>
                <div style='font-family:DM Mono;font-size:0.78rem;color:#8899aa;margin-bottom:0.5rem;'>FORMULA</div>
                <div class='code-block'>ŷ = β₀ + β₁·month + β₂·day + β₃·weekday + β₄·is_weekend</div>
            </div>
            <div style='margin-top:1rem;'>
                <div style='font-family:DM Mono;font-size:0.78rem;color:#00e5a0;'>✅ ADVANTAGES</div>
                <ul style='color:#c8d8e8;font-size:0.88rem;line-height:1.9;margin-top:0.4rem;'>
                    <li>Simple and fast to train</li>
                    <li>Easy to interpret</li>
                    <li>Works well when patterns are linear</li>
                </ul>
            </div>
            <div style='margin-top:0.5rem;'>
                <div style='font-family:DM Mono;font-size:0.78rem;color:#ff7b54;'>⚠️ LIMITATIONS</div>
                <ul style='color:#c8d8e8;font-size:0.88rem;line-height:1.9;margin-top:0.4rem;'>
                    <li>Cannot capture nonlinear patterns</li>
                    <li>Sensitive to outliers</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='section-card'>
            <div style='font-family:Syne;font-weight:800;font-size:1.2rem;color:#c084fc;margin-bottom:0.8rem;'>
                🌲 Random Forest Regressor
            </div>
            <p style='color:#c8d8e8;line-height:1.7;font-size:0.9rem;'>
                An <strong style='color:#f0f4ff;'>ensemble method</strong> that trains many decision trees on random subsets
                of data and averages their predictions. Captures complex non-linear interactions.
            </p>
            <div style='margin-top:1rem;'>
                <div style='font-family:DM Mono;font-size:0.78rem;color:#8899aa;margin-bottom:0.5rem;'>CONFIG</div>
                <div class='code-block'>n_estimators=100, random_state=42</div>
            </div>
            <div style='margin-top:1rem;'>
                <div style='font-family:DM Mono;font-size:0.78rem;color:#00e5a0;'>✅ ADVANTAGES</div>
                <ul style='color:#c8d8e8;font-size:0.88rem;line-height:1.9;margin-top:0.4rem;'>
                    <li>Captures nonlinear patterns</li>
                    <li>Handles feature interactions</li>
                    <li>Robust to outliers</li>
                </ul>
            </div>
            <div style='margin-top:0.5rem;'>
                <div style='font-family:DM Mono;font-size:0.78rem;color:#ff7b54;'>⚠️ LIMITATIONS</div>
                <ul style='color:#c8d8e8;font-size:0.88rem;line-height:1.9;margin-top:0.4rem;'>
                    <li>Computationally expensive</li>
                    <li>Harder to interpret</li>
                    <li>Can overfit if signal is weak</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Training code
    st.markdown("### 💻 Training Code")
    st.code("""
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Features and target
features = ['month', 'day', 'weekday', 'is_weekend']

x_train = train[features]
y_train = train['revenue']
x_test  = test[features]
y_test  = test['revenue']

# --- Linear Regression ---
lr = LinearRegression()
lr.fit(x_train, y_train)
lr_preds = lr.predict(x_test)

# --- Random Forest ---
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(x_train, y_train)
rf_preds = rf.predict(x_test)

# Evaluation
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
    """, language="python")

    # LR coefficients
    st.markdown("### 📊 Linear Regression — Learned Coefficients")
    coef_df = pd.DataFrame({
        "Feature": features,
        "Coefficient": lr_model.coef_.round(2),
    })
    fig_coef = go.Figure(go.Bar(
        x=coef_df["Feature"], y=coef_df["Coefficient"],
        marker_color=[COLORS["cyan"] if c > 0 else COLORS["orange"] for c in coef_df["Coefficient"]],
        text=coef_df["Coefficient"],
        textposition="outside",
        textfont=dict(family="DM Mono", color="#f0f4ff"),
    ))
    fig_coef.update_layout(
        **PLOT_LAYOUT,
        title=f"Learned Coefficients (Intercept: {lr_model.intercept_:.2f})",
        title_font=dict(family="Syne", size=16, color="#f0f4ff"),
        height=320,
    )
    st.plotly_chart(fig_coef, use_container_width=True)

    # RF feature importance
    st.markdown("### 🌲 Random Forest — Feature Importances")
    fi_df = pd.DataFrame({
        "Feature": features,
        "Importance": rf_model.feature_importances_.round(4),
    }).sort_values("Importance", ascending=True)

    fig_fi = go.Figure(go.Bar(
        x=fi_df["Importance"], y=fi_df["Feature"],
        orientation="h",
        marker_color=COLORS["purple"],
        text=fi_df["Importance"].round(3),
        textposition="outside",
        textfont=dict(family="DM Mono", color="#f0f4ff"),
    ))
    fig_fi.update_layout(
        **PLOT_LAYOUT,
        title="Feature Importances — Random Forest",
        title_font=dict(family="Syne", size=16, color="#f0f4ff"),
        height=280,
    )
    st.plotly_chart(fig_fi, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE: Results & Evaluation
# ════════════════════════════════════════════════════════════════════════════════
elif "Results" in page:
    if not data_loaded:
        st.error(f"Could not load datasets.\n\n`{load_error}`")
        st.stop()

    section_header("📊", "Results & Evaluation", "Model performance, error analysis, and prediction plots")

    # RMSE comparison
    col1, col2, col3 = st.columns([1, 1, 1.2])
    col1.metric("Linear Regression RMSE", f"R$ {lr_rmse:,.0f}", "Per-day avg error")
    col2.metric("Random Forest RMSE", f"R$ {rf_rmse:,.0f}", "Per-day avg error")
    winner = "Linear Regression" if lr_rmse < rf_rmse else "Random Forest"
    diff = abs(lr_rmse - rf_rmse)
    col3.metric("Winner", winner, f"By R$ {diff:,.0f} RMSE")

    st.markdown("<br>", unsafe_allow_html=True)

    # RMSE bar chart
    fig_rmse = go.Figure()
    fig_rmse.add_trace(go.Bar(
        x=["Linear Regression", "Random Forest"],
        y=[lr_rmse, rf_rmse],
        marker_color=[COLORS["cyan"], COLORS["purple"]],
        text=[f"R$ {lr_rmse:,.0f}", f"R$ {rf_rmse:,.0f}"],
        textposition="outside",
        textfont=dict(family="DM Mono", color="#f0f4ff", size=13),
        width=0.5,
    ))
    fig_rmse.add_hline(
        y=min(lr_rmse, rf_rmse),
        line_dash="dot", line_color="#fcd34d",
        annotation_text="Best model",
        annotation_font_color="#fcd34d",
    )
    fig_rmse.update_layout(
        **PLOT_LAYOUT,
        title="RMSE Comparison — Lower is Better",
        title_font=dict(family="Syne", size=16, color="#f0f4ff"),
        yaxis_title="RMSE (BRL)",
        height=350,
    )
    st.plotly_chart(fig_rmse, use_container_width=True)

    # Actual vs Predicted (LR)
    st.markdown("### 📈 Actual vs Predicted — Linear Regression")
    pred_df = pd.DataFrame({
        "date": test["date"].values,
        "Actual": y_test.values,
        "Linear Regression": lr_preds,
        "Random Forest": rf_preds,
    })

    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(
        x=pred_df["date"], y=pred_df["Actual"],
        name="Actual Revenue",
        line=dict(color=COLORS["cyan"], width=2),
        mode="lines",
    ))
    fig_pred.add_trace(go.Scatter(
        x=pred_df["date"], y=pred_df["Linear Regression"],
        name="LR Predicted",
        line=dict(color=COLORS["orange"], width=1.5, dash="dot"),
        mode="lines",
    ))
    fig_pred.update_layout(
        **PLOT_LAYOUT,
        title="Actual vs Linear Regression Predictions (Test Period)",
        title_font=dict(family="Syne", size=16, color="#f0f4ff"),
        yaxis_title="Revenue (BRL)",
        height=400,
    )
    st.plotly_chart(fig_pred, use_container_width=True)

    # Actual vs Predicted (RF)
    st.markdown("### 🌲 Actual vs Predicted — Random Forest")
    fig_pred_rf = go.Figure()
    fig_pred_rf.add_trace(go.Scatter(
        x=pred_df["date"], y=pred_df["Actual"],
        name="Actual Revenue",
        line=dict(color=COLORS["cyan"], width=2),
        mode="lines",
    ))
    fig_pred_rf.add_trace(go.Scatter(
        x=pred_df["date"], y=pred_df["Random Forest"],
        name="RF Predicted",
        line=dict(color=COLORS["purple"], width=1.5, dash="dot"),
        mode="lines",
    ))
    fig_pred_rf.update_layout(
        **PLOT_LAYOUT,
        title="Actual vs Random Forest Predictions (Test Period)",
        title_font=dict(family="Syne", size=16, color="#f0f4ff"),
        yaxis_title="Revenue (BRL)",
        height=400,
    )
    st.plotly_chart(fig_pred_rf, use_container_width=True)

    # Residuals
    st.markdown("### 🎯 Residual Analysis")
    pred_df["LR Residual"] = pred_df["Actual"] - pred_df["Linear Regression"]
    pred_df["RF Residual"] = pred_df["Actual"] - pred_df["Random Forest"]

    fig_res = make_subplots(rows=1, cols=2, subplot_titles=["LR Residuals", "RF Residuals"])
    fig_res.add_trace(go.Scatter(
        x=pred_df["date"], y=pred_df["LR Residual"],
        mode="lines", name="LR", line=dict(color=COLORS["orange"], width=1)
    ), row=1, col=1)
    fig_res.add_trace(go.Scatter(
        x=pred_df["date"], y=pred_df["RF Residual"],
        mode="lines", name="RF", line=dict(color=COLORS["purple"], width=1)
    ), row=1, col=2)
    fig_res.add_hline(y=0, line_color="#fcd34d", line_dash="dash")
    fig_res.update_layout(
        **PLOT_LAYOUT,
        title="Residuals (Actual − Predicted) Over Time",
        title_font=dict(family="Syne", size=16, color="#f0f4ff"),
        height=320,
        showlegend=False,
    )
    st.plotly_chart(fig_res, use_container_width=True)

    # Model comparison table
    st.markdown("### 📋 Model Comparison Summary")
    summary = pd.DataFrame({
        "Model": ["Linear Regression", "Random Forest"],
        "RMSE (BRL)": [f"{lr_rmse:,.0f}", f"{rf_rmse:,.0f}"],
        "Best For": ["Linear seasonal patterns", "Non-linear complex data"],
        "Interpretability": ["High", "Low"],
        "Training Speed": ["Fast", "Moderate"],
        "Winner": ["✅ Yes" if lr_rmse < rf_rmse else "", "✅ Yes" if rf_rmse < lr_rmse else ""],
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.markdown("""
    <div class='insight-card' style='margin-top:1rem;'>
        <strong>Why did Linear Regression win?</strong> The time-based features (month, day, weekday, is_weekend)
        have a <em>relatively linear</em> relationship with daily revenue. Random Forest's complexity
        offers no advantage when the underlying signal is straightforward — it may even overfit to noise.
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE: Business Insights
# ════════════════════════════════════════════════════════════════════════════════
elif "Business Insights" in page:
    if not data_loaded:
        st.error(f"Could not load datasets.\n\n`{load_error}`")
        st.stop()

    section_header("💡", "Business Insights", "Actionable findings from the data and model")

    # Key insights
    st.markdown("### 🔑 Key Findings")

    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                   7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    monthly_avg = daily_sales.groupby("month")["revenue"].mean()
    best_month_num = monthly_avg.idxmax()
    best_month = month_names[best_month_num]
    worst_month = month_names[monthly_avg.idxmin()]

    day_names = {0:"Monday",1:"Tuesday",2:"Wednesday",3:"Thursday",4:"Friday",5:"Saturday",6:"Sunday"}
    weekday_avg = daily_sales.groupby("weekday")["revenue"].mean()
    best_day = day_names[weekday_avg.idxmax()]
    worst_day = day_names[weekday_avg.idxmin()]

    weekend_avg = daily_sales[daily_sales["is_weekend"]==1]["revenue"].mean()
    weekday_avg_num = daily_sales[daily_sales["is_weekend"]==0]["revenue"].mean()
    pct_diff = ((weekday_avg_num - weekend_avg) / weekend_avg * 100)

    insights = [
        (f"📅 <strong>Best Month: {best_month}</strong>", f"Average daily revenue peaks in {best_month}, making it the highest-performing month of the year. Businesses should maximize inventory and marketing budgets entering this period."),
        (f"📉 <strong>Slowest Month: {worst_month}</strong>", f"{worst_month} consistently shows the lowest average daily revenue. Consider running promotions or targeted campaigns to offset the seasonal dip."),
        (f"🗓️ <strong>Best Sales Day: {best_day}</strong>", f"{best_day} generates the highest average revenue of all weekdays. Prime day for flash sales, email campaigns, and push notifications."),
        (f"😴 <strong>Slowest Day: {worst_day}</strong>", f"{worst_day} sees the lowest purchase activity. Consider lighter logistics operations and reduced ad spend on this day."),
        (f"💼 <strong>Weekdays Outperform Weekends by {pct_diff:.0f}%</strong>", f"Weekday average (R$ {weekday_avg_num:,.0f}) significantly exceeds weekend average (R$ {weekend_avg:,.0f}). This is common in B2C e-commerce — align campaigns and staffing to weekday peaks."),
        ("🤖 <strong>Linear Patterns Dominate</strong>", f"Linear Regression (RMSE R$ {lr_rmse:,.0f}) outperformed Random Forest (RMSE R$ {rf_rmse:,.0f}), confirming that time-based features have a simple, predictable relationship with revenue."),
        ("📦 <strong>Multi-Item Orders are Common</strong>", "13,984 duplicate order IDs in the items table represent multi-product purchases — basket size optimization could significantly lift average order revenue."),
    ]

    for title, body in insights:
        st.markdown(f"""
        <div class='insight-card'>
            {title}<br>
            <span style='color:#c8d8e8;font-size:0.9rem;'>{body}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Monthly revenue plot
    st.markdown("### 📊 Revenue by Month — Business Lens")
    monthly_full = daily_sales.groupby("month")["revenue"].agg(["mean","sum","std"]).reset_index()
    monthly_full.columns = ["month","avg","total","std"]
    monthly_full["month_name"] = monthly_full["month"].map(month_names)

    fig_biz = go.Figure()
    fig_biz.add_trace(go.Bar(
        x=monthly_full["month_name"], y=monthly_full["avg"],
        name="Avg Daily Revenue",
        marker_color=[COLORS["cyan"] if m == best_month_num else "rgba(0,212,255,0.35)" for m in monthly_full["month"]],
        error_y=dict(type="data", array=monthly_full["std"].tolist(), color="#fcd34d", thickness=1.5, width=4),
    ))
    fig_biz.update_layout(
        **PLOT_LAYOUT,
        title=f"Average Daily Revenue by Month (Peak: {best_month})",
        title_font=dict(family="Syne", size=16, color="#f0f4ff"),
        yaxis_title="Avg Daily Revenue (BRL)",
        xaxis_categoryorder="array",
        xaxis_categoryarray=list(month_names.values()),
    )
    st.plotly_chart(fig_biz, use_container_width=True)

    # Conclusion
    st.markdown("### 🏁 Conclusion")
    st.markdown("""
    <div class='section-card'>
        <div style='font-family:Syne;font-weight:700;font-size:1.1rem;color:#00d4ff;margin-bottom:1rem;'>Project Summary</div>
        <p style='color:#c8d8e8;line-height:1.8;'>
            This project demonstrates a complete end-to-end machine learning workflow applied to real-world
            Brazilian e-commerce data. Starting from raw transaction records, we performed data cleaning,
            feature engineering, exploratory analysis, model training, and evaluation.
        </p>
        <p style='color:#c8d8e8;line-height:1.8;'>
            The <strong style='color:#00d4ff;'>Linear Regression model emerged as the best performer</strong>, achieving a lower RMSE
            than Random Forest. This outcome reveals that the dominant revenue signal in this dataset is
            driven by simple, linear seasonal and calendar patterns — demonstrating that model complexity
            should always be validated against simpler baselines.
        </p>
        <div style='margin-top:1.2rem;display:flex;flex-wrap:wrap;gap:0.5rem;'>
            <span class='badge'>Data Cleaning</span>
            <span class='badge'>Feature Engineering</span>
            <span class='badge'>EDA</span>
            <span class='badge'>Time Series Split</span>
            <span class='badge'>Linear Regression</span>
            <span class='badge'>Random Forest</span>
            <span class='badge'>RMSE Evaluation</span>
            <span class='badge'>Business Insights</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
