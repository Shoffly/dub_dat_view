"""
OLX Supply Analysis - Dynamic Scatter Plot
Interactive Streamlit app for visualizing supply data from OLX listings

Features:
- Dynamic scatter plot with configurable X, Y axes
- Color and size dimensions
- Multi-filter support for all categorical columns
- Date range filtering
- Aggregate views (daily, weekly, monthly counts)
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from google.cloud import bigquery
from google.oauth2 import service_account
from datetime import datetime, timedelta
import numpy as np

# Page config with dark theme aesthetic
st.set_page_config(
    page_title="Dubizzle Supply Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a modern, distinctive look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Outfit:wght@300;400;600;700&display=swap');

    :root {
        --bg-primary: #0a0a0f;
        --bg-secondary: #12121a;
        --accent-cyan: #00d4ff;
        --accent-magenta: #ff00aa;
        --accent-yellow: #ffd700;
        --text-primary: #e8e8e8;
        --text-muted: #8888aa;
    }

    .stApp {
        background: linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 50%, #16213e 100%);
    }

    .main-title {
        font-family: 'Outfit', sans-serif;
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00d4ff, #ff00aa, #ffd700);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }

    .subtitle {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1rem;
        color: #8888aa;
        text-align: center;
        margin-bottom: 2rem;
    }

    .metric-container {
        background: linear-gradient(145deg, #1a1a2e, #12121a);
        border: 1px solid rgba(0, 212, 255, 0.2);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }

    .metric-value {
        font-family: 'Outfit', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        color: #00d4ff;
    }

    .metric-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: #8888aa;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .stSelectbox > div > div {
        background-color: #1a1a2e;
        border: 1px solid rgba(0, 212, 255, 0.3);
    }

    .stMultiSelect > div > div {
        background-color: #1a1a2e;
        border: 1px solid rgba(0, 212, 255, 0.3);
    }

    .sidebar .stSelectbox label, .sidebar .stMultiSelect label {
        font-family: 'JetBrains Mono', monospace;
        color: #00d4ff;
        font-size: 0.85rem;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #12121a 0%, #0a0a0f 100%);
        border-right: 1px solid rgba(0, 212, 255, 0.1);
    }

    .stButton > button {
        font-family: 'Outfit', sans-serif;
        background: linear-gradient(90deg, #00d4ff, #0088cc);
        color: #0a0a0f;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.75rem 2rem;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 212, 255, 0.4);
    }

    .section-header {
        font-family: 'Outfit', sans-serif;
        font-size: 1.5rem;
        font-weight: 600;
        color: #ff00aa;
        border-bottom: 2px solid rgba(255, 0, 170, 0.3);
        padding-bottom: 0.5rem;
        margin: 2rem 0 1rem 0;
    }

    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(0, 212, 255, 0.5), transparent);
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_bigquery_client():
    """Initialize BigQuery client with credentials"""
    try:
        credentials = service_account.Credentials.from_service_account_info(
            st.secrets["service_account"]
        )
    except (KeyError, FileNotFoundError):
        try:
            credentials = service_account.Credentials.from_service_account_file(
                'service_account.json'
            )
        except FileNotFoundError:
            st.error("❌ No BigQuery credentials found. Please add service_account.json or configure Streamlit secrets.")
            return None

    return bigquery.Client(credentials=credentials)


@st.cache_data(ttl=3600)
def load_data():
    """Load OLX listings data from BigQuery"""
    client = get_bigquery_client()
    if not client:
        return None

    query = """
    SELECT 
        id, 
        title, 
        transmission_type, 
        year, 
        Kilometers, 
        make, 
        model, 
        payment_options, 
        condition, 
        region, 
        price, 
        DATE(added_at) as added_at, 
        DATE(deactivated_at) as deactivated_at, 
        DATE_DIFF(DATE(deactivated_at), DATE(added_at), DAY) as days_to_deactivated
    FROM olx.listings 
    WHERE deactivated_at IS NOT NULL and DATE(added_at) >= '2025-10-01' 
    ORDER BY DATE(deactivated_at) DESC
    """

    try:
        df = client.query(query).to_dataframe()

        # Convert numeric columns to proper types
        numeric_cols = ['price', 'year', 'Kilometers', 'days_to_deactivated']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return None


def create_aggregate_data(df, time_granularity, group_by_col=None):
    """Create aggregated time series data"""
    df = df.copy()

    if time_granularity == "Daily":
        df['period'] = df['deactivated_at']
    elif time_granularity == "Weekly":
        df['period'] = pd.to_datetime(df['deactivated_at']).dt.to_period('W').dt.start_time
    elif time_granularity == "Monthly":
        df['period'] = pd.to_datetime(df['deactivated_at']).dt.to_period('M').dt.start_time

    if group_by_col and group_by_col != "None":
        agg_df = df.groupby(['period', group_by_col]).agg({
            'id': 'count',
            'price': 'mean',
            'days_to_deactivated': 'mean',
            'Kilometers': 'mean'
        }).reset_index()
        agg_df.columns = ['period', group_by_col, 'count', 'avg_price', 'avg_days_to_sell', 'avg_kilometers']
    else:
        agg_df = df.groupby('period').agg({
            'id': 'count',
            'price': 'mean',
            'days_to_deactivated': 'mean',
            'Kilometers': 'mean'
        }).reset_index()
        agg_df.columns = ['period', 'count', 'avg_price', 'avg_days_to_sell', 'avg_kilometers']

    return agg_df


def main():
    # Header
    st.markdown('<h1>Dubizzle Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<h3>// Dynamic visualization of deactivated listings</h3>', unsafe_allow_html=True)

    # Load data
    with st.spinner("🔄 Loading data from BigQuery..."):
        df = load_data()

    if df is None or df.empty:
        st.error("No data available. Please check your BigQuery connection.")
        return

    st.success(f"✅ Loaded {len(df):,} listings")

    # Sidebar - Filters and Configuration
    with st.sidebar:
        st.markdown("## 🎛️ Configuration")

        # View Mode
        view_mode = st.radio(
            "View Mode",
            ["Scatter Plot", "Time Series", "Distribution"],
            index=0
        )

        st.markdown("---")
        st.markdown("## 🔍 Filters")

        # Date range filter
        min_date = pd.to_datetime(df['deactivated_at']).min()
        max_date = pd.to_datetime(df['deactivated_at']).max()

        date_range = st.date_input(
            "Deactivation Date Range",
            value=(max_date - timedelta(days=90), max_date),
            min_value=min_date,
            max_value=max_date
        )

        # Categorical filters
        makes = ["All"] + sorted(df['make'].dropna().unique().tolist())
        selected_makes = st.multiselect("Make", makes, default=["All"])

        # Filter models based on selected makes
        if "All" in selected_makes:
            available_models = df['model'].dropna().unique().tolist()
        else:
            available_models = df[df['make'].isin(selected_makes)]['model'].dropna().unique().tolist()
        models = ["All"] + sorted(available_models)
        selected_models = st.multiselect("Model", models, default=["All"])

        regions = ["All"] + sorted(df['region'].dropna().unique().tolist())
        selected_regions = st.multiselect("Region", regions, default=["All"])

        transmissions = ["All"] + sorted(df['transmission_type'].dropna().unique().tolist())
        selected_transmissions = st.multiselect("Transmission", transmissions, default=["All"])

        conditions = ["All"] + sorted(df['condition'].dropna().unique().tolist())
        selected_conditions = st.multiselect("Condition", conditions, default=["All"])

        # Year range
        year_min_val = df['year'].dropna().min()
        year_max_val = df['year'].dropna().max()
        min_year = int(year_min_val) if pd.notna(year_min_val) else 2000
        max_year = int(year_max_val) if pd.notna(year_max_val) else 2025
        year_range = st.slider("Year Range", min_year, max_year, (min_year, max_year))

        # Price range
        price_min_val = df['price'].dropna().min()
        price_max_val = df['price'].dropna().max()
        min_price = int(price_min_val) if pd.notna(price_min_val) else 0
        max_price = int(min(price_max_val, 10000000)) if pd.notna(price_max_val) else 5000000
        price_range = st.slider(
            "Price Range",
            min_price,
            max_price,
            (min_price, max_price),
            format="%d EGP"
        )

        # Kilometers range
        km_max_val = df['Kilometers'].dropna().max()
        min_km = 0
        max_km = int(min(km_max_val, 500000)) if pd.notna(km_max_val) else 500000
        km_range = st.slider(
            "Kilometers Range",
            min_km,
            max_km,
            (min_km, max_km),
            format="%d km"
        )

    # Apply filters
    filtered_df = df.copy()

    # Date filter
    if len(date_range) == 2:
        filtered_df = filtered_df[
            (pd.to_datetime(filtered_df['deactivated_at']) >= pd.to_datetime(date_range[0])) &
            (pd.to_datetime(filtered_df['deactivated_at']) <= pd.to_datetime(date_range[1]))
            ]

    # Categorical filters
    if "All" not in selected_makes:
        filtered_df = filtered_df[filtered_df['make'].isin(selected_makes)]

    if "All" not in selected_models:
        filtered_df = filtered_df[filtered_df['model'].isin(selected_models)]

    if "All" not in selected_regions:
        filtered_df = filtered_df[filtered_df['region'].isin(selected_regions)]

    if "All" not in selected_transmissions:
        filtered_df = filtered_df[filtered_df['transmission_type'].isin(selected_transmissions)]

    if "All" not in selected_conditions:
        filtered_df = filtered_df[filtered_df['condition'].isin(selected_conditions)]

    # Numeric filters (handle NaN values)
    filtered_df = filtered_df[
        (filtered_df['year'].fillna(year_range[0]) >= year_range[0]) &
        (filtered_df['year'].fillna(year_range[1]) <= year_range[1])
        ]

    filtered_df = filtered_df[
        (filtered_df['price'].fillna(price_range[0]) >= price_range[0]) &
        (filtered_df['price'].fillna(price_range[1]) <= price_range[1])
        ]

    filtered_df = filtered_df[
        (filtered_df['Kilometers'].fillna(km_range[0]) >= km_range[0]) &
        (filtered_df['Kilometers'].fillna(km_range[1]) <= km_range[1])
        ]

    # Metrics row
    st.markdown('<p class="section-header">📈 Key Metrics</p>', unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Listings", f"{len(filtered_df):,}")

    with col2:
        avg_price = filtered_df['price'].mean()
        st.metric("Avg Price", f"{avg_price:,.0f} EGP" if pd.notna(avg_price) else "N/A")

    with col3:
        avg_days = filtered_df['days_to_deactivated'].mean()
        st.metric("Avg Days to Sell", f"{avg_days:.1f}" if pd.notna(avg_days) else "N/A")

    with col4:
        avg_km = filtered_df['Kilometers'].mean()
        st.metric("Avg Kilometers", f"{avg_km:,.0f}" if pd.notna(avg_km) else "N/A")

    with col5:
        unique_makes = filtered_df['make'].nunique()
        st.metric("Unique Makes", f"{unique_makes}")

    st.markdown("---")

    # View-specific controls and visualization
    if view_mode == "Scatter Plot":
        st.markdown('<p class="section-header">⚙️ Scatter Plot Configuration</p>', unsafe_allow_html=True)

        # Numeric columns for axes
        numeric_cols = ['price', 'year', 'Kilometers', 'days_to_deactivated']
        categorical_cols = ['make', 'model', 'region', 'transmission_type', 'condition', 'payment_options']

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            x_axis = st.selectbox("X Axis", numeric_cols, index=0)

        with col2:
            y_axis = st.selectbox("Y Axis", numeric_cols, index=2)

        with col3:
            color_by = st.selectbox("Color By", ["None"] + categorical_cols, index=1)

        with col4:
            size_by = st.selectbox("Size By", ["None"] + numeric_cols, index=0)

        # Additional options
        col5, col6 = st.columns(2)

        with col5:
            max_sample = min(len(filtered_df), 50000)
            if max_sample <= 100:
                sample_size = max_sample
                st.info(f"Sample size: {sample_size} (all data)")
            else:
                sample_size = st.slider(
                    "Sample Size (for performance)",
                    min_value=100,
                    max_value=max_sample,
                    value=min(max_sample, 5000),
                    step=100
                )

        with col6:
            opacity = st.slider("Point Opacity", 0.1, 1.0, 0.6, 0.1)

        # Sample data for performance
        if len(filtered_df) > sample_size:
            plot_df = filtered_df.sample(n=sample_size, random_state=42)
            st.info(f"📊 Showing {sample_size:,} samples out of {len(filtered_df):,} total listings")
        else:
            plot_df = filtered_df

        # Create scatter plot
        st.markdown('<p class="section-header">📊 Scatter Plot</p>', unsafe_allow_html=True)

        fig_kwargs = {
            'x': x_axis,
            'y': y_axis,
            'opacity': opacity,
            'hover_data': ['title', 'make', 'model', 'year', 'price', 'region'],
            'template': 'plotly_dark'
        }

        if color_by != "None":
            fig_kwargs['color'] = color_by

        if size_by != "None":
            fig_kwargs['size'] = size_by
            fig_kwargs['size_max'] = 20

        fig = px.scatter(plot_df, **fig_kwargs)

        # Customize layout
        fig.update_layout(
            plot_bgcolor='rgba(10, 10, 15, 0.8)',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            font=dict(family="JetBrains Mono", color="#e8e8e8"),
            title=dict(
                text=f"{y_axis.replace('_', ' ').title()} vs {x_axis.replace('_', ' ').title()}",
                font=dict(size=20, color="#00d4ff")
            ),
            xaxis=dict(
                title=x_axis.replace('_', ' ').title(),
                gridcolor='rgba(0, 212, 255, 0.1)',
                zerolinecolor='rgba(0, 212, 255, 0.2)'
            ),
            yaxis=dict(
                title=y_axis.replace('_', ' ').title(),
                gridcolor='rgba(0, 212, 255, 0.1)',
                zerolinecolor='rgba(0, 212, 255, 0.2)'
            ),
            legend=dict(
                bgcolor='rgba(18, 18, 26, 0.8)',
                bordercolor='rgba(0, 212, 255, 0.3)',
                borderwidth=1
            ),
            height=600
        )

        st.plotly_chart(fig, use_container_width=True)

    elif view_mode == "Time Series":
        st.markdown('<p class="section-header">⚙️ Time Series Configuration</p>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            time_granularity = st.selectbox("Time Granularity", ["Daily", "Weekly", "Monthly"], index=1)

        with col2:
            y_metric = st.selectbox(
                "Y Axis Metric",
                ["count", "avg_price", "avg_days_to_sell", "avg_kilometers"],
                index=0,
                format_func=lambda x: x.replace('_', ' ').title()
            )

        with col3:
            group_by = st.selectbox(
                "Group By",
                ["None", "make", "region", "transmission_type", "condition"],
                index=0
            )

        # Create aggregated data
        agg_df = create_aggregate_data(filtered_df, time_granularity, group_by if group_by != "None" else None)

        st.markdown('<p class="section-header">📈 Time Series</p>', unsafe_allow_html=True)

        if group_by != "None":
            # Limit to top N categories for readability
            top_n = st.slider("Show Top N Categories", 3, 15, 8)
            top_cats = agg_df.groupby(group_by)['count'].sum().nlargest(top_n).index.tolist()
            agg_df = agg_df[agg_df[group_by].isin(top_cats)]

            fig = px.line(
                agg_df,
                x='period',
                y=y_metric,
                color=group_by,
                markers=True,
                template='plotly_dark'
            )
        else:
            fig = px.line(
                agg_df,
                x='period',
                y=y_metric,
                markers=True,
                template='plotly_dark'
            )
            fig.update_traces(line_color='#00d4ff', marker_color='#ff00aa')

        fig.update_layout(
            plot_bgcolor='rgba(10, 10, 15, 0.8)',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            font=dict(family="JetBrains Mono", color="#e8e8e8"),
            title=dict(
                text=f"{y_metric.replace('_', ' ').title()} Over Time ({time_granularity})",
                font=dict(size=20, color="#00d4ff")
            ),
            xaxis=dict(
                title="Date",
                gridcolor='rgba(0, 212, 255, 0.1)'
            ),
            yaxis=dict(
                title=y_metric.replace('_', ' ').title(),
                gridcolor='rgba(0, 212, 255, 0.1)'
            ),
            legend=dict(
                bgcolor='rgba(18, 18, 26, 0.8)',
                bordercolor='rgba(0, 212, 255, 0.3)',
                borderwidth=1
            ),
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        # Show data table
        with st.expander("📋 View Aggregated Data"):
            st.dataframe(agg_df, use_container_width=True)

    elif view_mode == "Distribution":
        st.markdown('<p class="section-header">⚙️ Distribution Configuration</p>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            dist_type = st.selectbox("Chart Type", ["Histogram", "Box Plot", "Violin Plot"])

        with col2:
            dist_column = st.selectbox(
                "Variable",
                ['price', 'year', 'Kilometers', 'days_to_deactivated'],
                index=0
            )

        with col3:
            split_by = st.selectbox(
                "Split By",
                ["None", "make", "region", "transmission_type", "condition"],
                index=0
            )

        st.markdown('<p class="section-header">📊 Distribution</p>', unsafe_allow_html=True)

        plot_df = filtered_df.copy()

        # Limit categories if splitting
        if split_by != "None":
            top_cats = plot_df[split_by].value_counts().nlargest(10).index.tolist()
            plot_df = plot_df[plot_df[split_by].isin(top_cats)]

        if dist_type == "Histogram":
            if split_by != "None":
                fig = px.histogram(
                    plot_df,
                    x=dist_column,
                    color=split_by,
                    barmode='overlay',
                    opacity=0.7,
                    template='plotly_dark'
                )
            else:
                fig = px.histogram(
                    plot_df,
                    x=dist_column,
                    template='plotly_dark'
                )
                fig.update_traces(marker_color='#00d4ff')

        elif dist_type == "Box Plot":
            if split_by != "None":
                fig = px.box(
                    plot_df,
                    x=split_by,
                    y=dist_column,
                    color=split_by,
                    template='plotly_dark'
                )
            else:
                fig = px.box(
                    plot_df,
                    y=dist_column,
                    template='plotly_dark'
                )

        else:  # Violin Plot
            if split_by != "None":
                fig = px.violin(
                    plot_df,
                    x=split_by,
                    y=dist_column,
                    color=split_by,
                    box=True,
                    template='plotly_dark'
                )
            else:
                fig = px.violin(
                    plot_df,
                    y=dist_column,
                    box=True,
                    template='plotly_dark'
                )

        fig.update_layout(
            plot_bgcolor='rgba(10, 10, 15, 0.8)',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            font=dict(family="JetBrains Mono", color="#e8e8e8"),
            title=dict(
                text=f"{dist_column.replace('_', ' ').title()} Distribution",
                font=dict(size=20, color="#00d4ff")
            ),
            xaxis=dict(gridcolor='rgba(0, 212, 255, 0.1)'),
            yaxis=dict(gridcolor='rgba(0, 212, 255, 0.1)'),
            legend=dict(
                bgcolor='rgba(18, 18, 26, 0.8)',
                bordercolor='rgba(0, 212, 255, 0.3)',
                borderwidth=1
            ),
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

    # Data explorer section
    st.markdown("---")
    st.markdown('<p class="section-header">🔎 Data Explorer</p>', unsafe_allow_html=True)

    with st.expander("View Filtered Data"):
        # Column selector
        available_cols = filtered_df.columns.tolist()
        selected_cols = st.multiselect(
            "Select Columns to Display",
            available_cols,
            default=['title', 'make', 'model', 'year', 'price', 'region', 'days_to_deactivated']
        )

        if selected_cols:
            st.dataframe(
                filtered_df[selected_cols].head(500),
                use_container_width=True
            )

            # Download option
            csv = filtered_df[selected_cols].to_csv(index=False)
            st.download_button(
                "📥 Download Filtered Data (CSV)",
                csv,
                file_name=f"olx_supply_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #8888aa; font-family: JetBrains Mono; font-size: 0.8rem;">'
        f'Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | Data source: OLX Listings'
        '</p>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()

