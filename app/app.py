import streamlit as st
import pandas as pd
import plotly.express as px
import os
import numpy as np
import json
import sys

# ============================
# DATABASE IMPORT
# ============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, 'database'))

try:
    from db import register_user, authenticate_user, get_all_users
except ImportError:
    st.error("❌ database/db.py file nahi mili.")
    st.stop()

PARQUET = os.path.join(BASE_DIR, 'parquet_data')
PY_OUT = os.path.join(BASE_DIR, 'python_pipeline')

st.set_page_config(page_title="DineIQ Analytics", layout="wide", page_icon="🍽️")

# ============================================================
# ROLE-BASED ACCESS CONTROL CONFIGURATION (SRS Page 28, 32)
# ============================================================
ROLE_PERMISSIONS = {
    'analyst': {
        'pages': [
            "🏠 Executive Dashboard",
            "📊 Menu Intelligence",
            "👥 Customer Intelligence"
        ],
        'can_download': False,
        'can_view_users': False,
        'can_view_audit': False,
        'can_run_whatif': False,
        'description': "Data Analyst — Read-only access to core dashboards"
    },
    'manager': {
        'pages': [
            "🏠 Executive Dashboard",
            "📊 Menu Intelligence",
            "👥 Customer Intelligence",
            "🗑️ Wastage Dashboard",
            "📈 Forecast Dashboard",
            "💰 Price & Promotion",
            "⏰ Peak Period Analysis",
            "🐢 Slow-Moving Dishes"
        ],
        'can_download': True,
        'can_view_users': False,
        'can_view_audit': False,
        'can_run_whatif': False,
        'description': "Restaurant Manager — Branch-level analytics + downloads"
    },
    'regional_manager': {
        'pages': [
            "🏠 Executive Dashboard",
            "📊 Menu Intelligence",
            "👥 Customer Intelligence",
            "🗑️ Wastage Dashboard",
            "📈 Forecast Dashboard",
            "💰 Price & Promotion",
            "📍 Location Comparison",
            "⏰ Peak Period Analysis",
            "🐢 Slow-Moving Dishes",
            "⚠️ Churn Risk",
            "🎁 Bundle Recommendations"
        ],
        'can_download': True,
        'can_view_users': False,
        'can_view_audit': False,
        'can_run_whatif': True,
        'description': "Regional Manager — Multi-location intelligence + what-if"
    },
    'admin': {
        'pages': [
            "🏠 Executive Dashboard",
            "📊 Menu Intelligence",
            "👥 Customer Intelligence",
            "🗑️ Wastage Dashboard",
            "📈 Forecast Dashboard",
            "💰 Price & Promotion",
            "📍 Location Comparison",
            "🔄 Dual-Pipeline Comparison",
            "🔮 What-If Analysis",
            "⏰ Peak Period Analysis",
            "🐢 Slow-Moving Dishes",
            "⚠️ Churn Risk",
            "🎁 Bundle Recommendations",
            "🎯 ML Model Comparison"
        ],
        'can_download': True,
        'can_view_users': True,
        'can_view_audit': True,
        'can_run_whatif': True,
        'description': "Administrator — Full system access"
    }
}

# ============================
# LOGIN / REGISTER
# ============================
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_role' not in st.session_state:
    st.session_state.user_role = None
if 'username' not in st.session_state:
    st.session_state.username = None

if not st.session_state.logged_in:
    st.markdown("""
    <style>
        .stApp { background: linear-gradient(135deg, #0E1117 0%, #1A1F2E 100%); }
        h1, h3 { color: #00D9FF !important; text-align: center; text-shadow: 0 0 20px rgba(0,217,255,0.3); }
        div[data-testid="stTextInput"] input {
            background: #1A1F2E; color: #FFFFFF;
            border: 1px solid #00D9FF55; border-radius: 8px;
        }
        .stButton > button {
            background: linear-gradient(90deg, #00D9FF 0%, #0088CC 100%);
            color: #0E1117; font-weight: bold;
            border-radius: 8px; border: none; width: 100%;
        }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        st.markdown("# 🍽️ DineIQ Analytics")
        st.markdown("### 🔐 Secure Access")
        st.markdown("---")
        
        tab1, tab2 = st.tabs(["🔑 Login", "📝 Register"])
        
        with tab1:
            username = st.text_input("👤 Username", key="login_user")
            password = st.text_input("🔑 Password", type="password", key="login_pass")
            if st.button("🚀 Login", key="login_btn"):
                if username and password:
                    success, user_info = authenticate_user(username, password)
                    if success:
                        st.session_state.logged_in = True
                        st.session_state.user_role = user_info['role']
                        st.session_state.username = user_info['username']
                        st.rerun()
                    else:
                        st.error("❌ Invalid credentials")
            st.info("**Demo Credentials:**\n- Admin: `admin` / `dineiq123`\n- Analyst: `analyst` / `analyst123`")
        
        with tab2:
            new_user = st.text_input("👤 New Username", key="reg_user")
            new_pass = st.text_input("🔑 Password (min 6)", type="password", key="reg_pass")
            role = st.selectbox("👔 Role", ['analyst', 'manager', 'regional_manager', 'admin'], key="reg_role")
            if st.button("✅ Register", key="reg_btn"):
                if len(new_user) < 3:
                    st.error("Username must be 3+ characters")
                elif len(new_pass) < 6:
                    st.error("Password must be 6+ characters")
                else:
                    success, msg = register_user(new_user, new_pass, role)
                    if success:
                        st.success(f"✅ {msg}")
                    else:
                        st.error(f"❌ {msg}")
    
    st.stop()

# ============================
# DASHBOARD CSS
# ============================
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0E1117 0%, #1A1F2E 100%); }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0A0E14 0%, #131720 100%);
        border-right: 1px solid #00D9FF33;
    }
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1A1F2E 0%, #232A3D 100%);
        border: 1px solid #00D9FF55;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(0, 217, 255, 0.1);
    }
    div[data-testid="stMetricLabel"] {
        color: #00D9FF !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        text-transform: uppercase;
    }
    div[data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-size: 26px !important;
        font-weight: 700 !important;
    }
    h1, h2, h3 { color: #00D9FF !important; text-shadow: 0 0 20px rgba(0, 217, 255, 0.3); }
    .stAlert { border-radius: 10px; border-left: 4px solid #00D9FF; }
    .stButton > button {
        background: linear-gradient(90deg, #00D9FF 0%, #0088CC 100%);
        color: #0E1117; font-weight: bold; border-radius: 8px; border: none;
    }
    .stDownloadButton > button {
        background: linear-gradient(90deg, #00D9FF 0%, #0088CC 100%);
        color: #0E1117; font-weight: bold; border-radius: 8px;
    }
    hr { border-color: #00D9FF33; }
    .role-badge {
        display: inline-block;
        padding: 4px 12px;
        background: linear-gradient(90deg, #00D9FF 0%, #0088CC 100%);
        color: #0E1117;
        border-radius: 12px;
        font-size: 11px;
        font-weight: bold;
        text-transform: uppercase;
    }
</style>
""", unsafe_allow_html=True)

# ============================
# APPLY ROLE PERMISSIONS
# ============================
role = st.session_state.user_role
permissions = ROLE_PERMISSIONS.get(role, ROLE_PERMISSIONS['analyst'])

# ============================
# SIDEBAR
# ============================
st.sidebar.title("🍽️ DineIQ Analytics")
st.sidebar.caption("Restaurant Intelligence Platform")

st.sidebar.markdown(f"### 👤 {st.session_state.username}")
st.sidebar.markdown(f'<span class="role-badge">{role.replace("_", " ").upper()}</span>', unsafe_allow_html=True)
st.sidebar.caption(permissions['description'])
st.sidebar.markdown("---")

# Role-based navigation
page = st.sidebar.radio("📌 Navigation", permissions['pages'])

# Admin-only: All Users
if permissions['can_view_users']:
    with st.sidebar.expander("👥 All Users"):
        try:
            users = get_all_users()
            for u in users:
                st.caption(f"• {u[1]} ({u[2]})")
        except:
            st.caption("No data")

# Admin-only: Audit Trail
if permissions['can_view_audit']:
    with st.sidebar.expander("📋 Audit Trail"):
        try:
            with open(f'{PARQUET}/audit_trail.json') as f:
                audit = json.load(f)
            st.caption(f"Session: {audit['session_id']}")
            for action in audit['actions']:
                st.caption(f"• {action['action']}")
        except:
            st.caption("No audit data")

st.sidebar.markdown("---")
if st.sidebar.button("🚪 Logout"):
    st.session_state.logged_in = False
    st.session_state.user_role = None
    st.session_state.username = None
    st.rerun()

st.sidebar.caption("v1.0 | Spark + Python")

# ============================
# LOAD DATA
# ============================
@st.cache_data
def load_data():
    d = {}
    files = [
        ('menu', f'{PARQUET}/menu_classified/'),
        ('orders', f'{PARQUET}/orders/'),
        ('seg', f'{PARQUET}/customer_segments/'),
        ('wastage', f'{PARQUET}/wastage_analysis/'),
        ('price', f'{PARQUET}/price_sensitivity/'),
        ('location', f'{PARQUET}/location_analysis/'),
        ('recs', f'{PARQUET}/recommendations/'),
        ('promo_traps', f'{PARQUET}/promo_traps/'),
        ('peak_hours', f'{PARQUET}/peak_hours/'),
        ('peak_days', f'{PARQUET}/peak_days/'),
        ('slow_items', f'{PARQUET}/slow_moving_items/'),
        ('churn', f'{PARQUET}/churn_risk/'),
        ('bundles', f'{PARQUET}/bundle_recommendations/'),
    ]
    for key, path in files:
        try:
            d[key] = pd.read_parquet(path)
        except Exception:
            d[key] = None
    
    try: d['forecast'] = pd.read_csv(f'{PY_OUT}/demand_predictions_python.csv')
    except: d['forecast'] = None
    try: d['comparison'] = pd.read_csv(f'{PY_OUT}/comparison_report.csv')
    except: d['comparison'] = None
    try:
        with open(f'{PARQUET}/ml_models_comparison.json') as f:
            d['ml_models'] = json.load(f)
    except: d['ml_models'] = None
    
    return d

data = load_data()

# ============================
# GLOBAL FILTERS
# ============================
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Global Filters")

if data.get('orders') is not None:
    orders = data['orders'].copy()
    orders['order_date'] = pd.to_datetime(orders['order_date'])
    
    date_min = orders['order_date'].min().date()
    date_max = orders['order_date'].max().date()
    date_range = st.sidebar.date_input("📅 Date Range", [date_min, date_max])
    
    locations = ['All'] + sorted(orders['location_id'].dropna().unique().tolist())
    sel_location = st.sidebar.selectbox("📍 Location", locations)
    
    channels = ['All'] + sorted(orders['channel'].dropna().unique().tolist())
    sel_channel = st.sidebar.selectbox("🛒 Channel", channels)
    
    filtered = orders.copy()
    if len(date_range) == 2:
        filtered = filtered[(filtered['order_date'].dt.date >= date_range[0]) & 
                            (filtered['order_date'].dt.date <= date_range[1])]
    if sel_location != 'All':
        filtered = filtered[filtered['location_id'] == sel_location]
    if sel_channel != 'All':
        filtered = filtered[filtered['channel'] == sel_channel]
else:
    filtered = None

# ============================
# PAGE 1: EXECUTIVE DASHBOARD
# ============================
if page == "🏠 Executive Dashboard":
    st.title("🏠 Executive Dashboard")
    st.caption(f"Logged in as: **{st.session_state.username}** ({role})")
    
    if filtered is not None and len(filtered) > 0:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("💰 Total Revenue", f"${filtered['total_amount'].sum():,.0f}")
        c2.metric("🛒 Total Orders", f"{len(filtered):,}")
        c3.metric("📊 Avg Order Value", f"${filtered['total_amount'].mean():.2f}")
        c4.metric("👥 Customers", f"{filtered['customer_id'].nunique():,}")
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            daily = filtered.groupby(filtered['order_date'].dt.date)['total_amount'].sum().reset_index()
            daily.columns = ['date', 'revenue']
            fig = px.line(daily, x='date', y='revenue', title='📈 Revenue Trend')
            fig.update_traces(line_color='#00D9FF')
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            channel = filtered.groupby('channel')['total_amount'].sum().reset_index()
            fig = px.pie(channel, names='channel', values='total_amount', title='🛒 Channel Mix')
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig, use_container_width=True)

# ============================
# PAGE 2: MENU INTELLIGENCE
# ============================
elif page == "📊 Menu Intelligence":
    st.title("📊 Menu Intelligence Dashboard")
    
    if data.get('menu') is not None:
        menu = data['menu']
        col1, col2 = st.columns(2)
        with col1:
            class_filter = st.multiselect("Filter by Class", menu['performance_class'].unique(), default=menu['performance_class'].unique())
        with col2:
            cat_filter = st.multiselect("Filter by Category", menu['category'].unique(), default=menu['category'].unique())
        
        menu_f = menu[(menu['performance_class'].isin(class_filter)) & (menu['category'].isin(cat_filter))]
        
        st.subheader("🏆 Top 10 Profitable Items")
        top = menu_f.nlargest(10, 'profit')[['item_name','qty_sold','revenue','profit','profit_pct','performance_class']]
        st.dataframe(top, use_container_width=True)
        
        if permissions['can_download']:
            st.download_button("📥 Download Top Items", top.to_csv(index=False), "top_items.csv", "text/csv")
        else:
            st.info("🔒 Download not available for your role")
        
        st.subheader("💡 Recommendations")
        hidden = menu_f[menu_f['performance_class'] == 'Hidden Opportunity'].nlargest(5, 'profit_pct')
        for _, r in hidden.iterrows():
            st.success(f"**Promote {r['item_name']}** — {r['profit_pct']}% margin")

# ============================
# PAGE 3: CUSTOMER INTELLIGENCE
# ============================
elif page == "👥 Customer Intelligence":
    st.title("👥 Customer Intelligence Dashboard")
    
    if data.get('seg') is not None:
        seg = data['seg']
        col1, col2 = st.columns(2)
        with col1:
            seg_counts = seg.groupby('prediction').size().reset_index(name='count')
            fig = px.bar(seg_counts, x='prediction', y='count', title='Customers per Segment')
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.scatter(seg, x='recency', y='monetary', color='prediction', size='frequency', title='RFM Scatter')
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig, use_container_width=True)

# ============================
# PAGE 4: WASTAGE
# ============================
elif page == "🗑️ Wastage Dashboard":
    st.title("🗑️ Wastage Dashboard")
    if data.get('wastage') is not None:
        w = data['wastage']
        c1, c2 = st.columns(2)
        c1.metric("💰 Total Wastage", f"${w['total_wastage_cost'].sum():,.2f}")
        c2.metric("⚠️ High Wastage Items", f"{len(w[w['wastage_pct'] > 10])}")
        
        top_w = w.nlargest(10, 'total_wastage_cost')[['item_name','total_wasted_qty','total_wastage_cost','wastage_pct']]
        st.dataframe(top_w, use_container_width=True)
        
        if permissions['can_download']:
            st.download_button("📥 Download Wastage Report", top_w.to_csv(index=False), "wastage.csv", "text/csv")

# ============================
# PAGE 5: FORECAST
# ============================
elif page == "📈 Forecast Dashboard":
    st.title("📈 Demand Forecast Dashboard")
    if data.get('forecast') is not None:
        fc = data['forecast']
        fc['date'] = pd.to_datetime(fc['date'])
        fig = px.line(fc, x='date', y=['demand','predicted'], title='Actual vs Predicted')
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig, use_container_width=True)

# ============================
# PAGE 6: PRICE & PROMO
# ============================
elif page == "💰 Price & Promotion":
    st.title("💰 Price & Promotion Intelligence")
    tab1, tab2 = st.tabs(["💵 Price Sensitivity", "🎯 Promotion Traps"])
    with tab1:
        if data.get('price') is not None:
            st.dataframe(data['price'].head(20), use_container_width=True)
    with tab2:
        if data.get('promo_traps') is not None:
            st.dataframe(data['promo_traps'], use_container_width=True)

# ============================
# PAGE 7: LOCATION COMPARISON
# ============================
elif page == "📍 Location Comparison":
    st.title("📍 Multi-Location Intelligence")
    if data.get('location') is not None:
        loc = data['location']
        top_loc = loc.nlargest(10, 'revenue')[['location_id','revenue','total_orders','avg_order_value','profit_margin_pct']]
        st.dataframe(top_loc, use_container_width=True)
        fig = px.bar(top_loc, x='location_id', y='revenue', color='profit_margin_pct', title='Revenue by Location')
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig, use_container_width=True)

# ============================
# PAGE 8: DUAL-PIPELINE
# ============================
elif page == "🔄 Dual-Pipeline Comparison":
    st.title("🔄 Spark vs Python Comparison")
    if data.get('comparison') is not None:
        comp = data['comparison']
        total = len(comp)
        matches = int(comp['match'].sum())
        pct = (matches/total)*100 if total > 0 else 0
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Records", f"{total:,}")
        c2.metric("Matches", f"{matches:,}")
        c3.metric("Agreement %", f"{pct:.2f}%")
        st.dataframe(comp.head(20), use_container_width=True)

# ============================
# PAGE 9: WHAT-IF
# ============================
elif page == "🔮 What-If Analysis":
    st.title("🔮 What-If Scenario Analysis")
    st.info("⚠️ Simulated estimates — NOT actual results.")
    if data.get('menu') is not None:
        menu = data['menu']
        item_name = st.selectbox("Select Item", menu['item_name'].unique())
        item = menu[menu['item_name'] == item_name].iloc[0]
        price_change = st.slider("Price Change %", -50, 50, 0)
        new_price = item['base_price'] * (1 + price_change/100)
        c1, c2, c3 = st.columns(3)
        c1.metric("Current Price", f"${item['base_price']:.2f}")
        c2.metric("New Price", f"${new_price:.2f}")
        c3.metric("Qty Sold", f"{item['qty_sold']:,}")

# ============================
# PAGE 10: PEAK PERIOD
# ============================
elif page == "⏰ Peak Period Analysis":
    st.title("⏰ Peak Period Analysis")
    if data.get('peak_hours') is not None:
        peak_hours = data['peak_hours']
        fig = px.bar(peak_hours, x='hour', y='order_count', title='Orders by Hour')
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig, use_container_width=True)

# ============================
# PAGE 11: SLOW-MOVING
# ============================
elif page == "🐢 Slow-Moving Dishes":
    st.title("🐢 Slow-Moving Dish Detection")
    if data.get('slow_items') is not None:
        st.dataframe(data['slow_items'].head(30), use_container_width=True)

# ============================
# PAGE 12: CHURN
# ============================
elif page == "⚠️ Churn Risk":
    st.title("⚠️ Customer Churn Risk")
    if data.get('churn') is not None:
        churn = data['churn']
        c1, c2 = st.columns(2)
        c1.metric("🔴 High Risk", f"{(churn['churn_risk']=='High Risk').sum():,}")
        c2.metric("🟢 Active", f"{(churn['churn_risk']=='Active').sum():,}")

# ============================
# PAGE 13: BUNDLES
# ============================
elif page == "🎁 Bundle Recommendations":
    st.title("🎁 Bundle Recommendations")
    if data.get('bundles') is not None:
        st.dataframe(data['bundles'].head(30), use_container_width=True)

# ============================
# PAGE 14: ML MODELS
# ============================
elif page == "🎯 ML Model Comparison":
    st.title("🎯 ML Model Comparison")
    if data.get('ml_models') is not None:
        ml = data['ml_models']
        models_df = pd.DataFrame(ml['models']).T.reset_index()
        models_df.columns = ['Model', 'Accuracy', 'F1', 'Precision', 'Recall']
        st.dataframe(models_df, use_container_width=True)
        st.success(f"🏆 Best Model: {ml['best']} | Version: {ml['version']}")

# ============================
# FOOTER
# ============================
st.sidebar.markdown("---")
st.sidebar.caption("© 2026 DineIQ Analytics")