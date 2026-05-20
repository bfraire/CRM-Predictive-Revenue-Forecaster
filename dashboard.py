# Import necessary libraries
import streamlit as st
import pandas as pd
import altair as alt
from datetime import date

# Page configuration: title, icon, and layout
st.set_page_config(
    page_title="CRM Revenue Forecast Dashboard",
    layout="wide"
)

st.markdown("""
<style>
.metric-card {
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 16px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    background-color: #f9f9f9;
}
.metric-label {
    font-size: 1em;
    color: #666666;
    font-weight: 500;
}
.metric-number {
    font-size: 4em;
    font-weight: bold;
    color: #0f4c81;
    margin-top: 4px;
}
.metric-number-neutral {
    font-size: 4em;
    font-weight: bold;
    color: #1976D2;
    margin-top: 4px;
}
</style>
""", unsafe_allow_html=True)

# Load Data with caching to improve performance on filter interactions
@st.cache_data
def load_data():
    """
    Load final_combined_forecast.csv and return a clean dataframe.
    """
    df = pd.read_csv('final_combined_forecast.csv', parse_dates=['close_date'])
    df['close_date'] = pd.to_datetime(df['close_date'])
    return df

combined_df = load_data()

# Build filter options based on data
start_date_initial    = combined_df['close_date'].min().date()
end_date_initial      = combined_df['close_date'].max().date()
products_list         = ["All"] + sorted(combined_df['product'].dropna().unique())
office_locations_list = ["All"] + sorted(combined_df['office_location'].dropna().unique())
deal_types_list       = ["All"] + sorted(combined_df['type'].dropna().unique())
chart_list            = ["Total Revenue Trend", "Revenue by Day", "Top Products"]

# Session state initialization for filters to maintain selections across interactions
if 'selected_start_date'     not in st.session_state:
    st.session_state.selected_start_date     = start_date_initial
if 'selected_end_date'       not in st.session_state:
    st.session_state.selected_end_date       = end_date_initial
if 'selected_product'        not in st.session_state:
    st.session_state.selected_product        = "All"
if 'selected_office_location' not in st.session_state:
    st.session_state.selected_office_location = "All"
if 'selected_deal_type'      not in st.session_state:
    st.session_state.selected_deal_type      = "All"
if 'selected_chart'          not in st.session_state:
    st.session_state.selected_chart          = chart_list[0]

# Data processing function that applies all filters and prepares KPIs, charts, and details table
def update_data(start_date, end_date, product, office_location, deal_type):
    """
    Applies all active filters to the combined dataframe and returns:
      - Three KPI strings for the metric cards
      - Three chart-ready dataframes
      - A details table dataframe
    """
    df = combined_df.copy()

    # ── Date filter ───────────────────────────────────────────────────────────
    df = df[
        (df['close_date'].dt.date >= start_date) &
        (df['close_date'].dt.date <= end_date)
    ]

    # ── Segment filters ───────────────────────────────────────────────────────
    if product        != "All": df = df[df['product']         == product]
    if office_location!= "All": df = df[df['office_location'] == office_location]
    if deal_type      != "All": df = df[df['type']            == deal_type]

    # ── KPI calculations ──────────────────────────────────────────────────────
    if not df.empty:
        total_revenue  = df['total_revenue'].sum()
        total_deals    = df['total_deals'].sum()
        avg_deal_size  = (total_revenue / total_deals) if total_deals > 0 else 0

        total_rev_str  = f"${total_revenue:,.0f}"
        total_deals_str= f"{total_deals:,.0f}"
        avg_deal_str   = f"${avg_deal_size:,.0f}"
    else:
        total_rev_str  = "--"
        total_deals_str= "--"
        avg_deal_str   = "--"

    # ── Chart 1: Monthly revenue trend ───────────────────────────────────────
    combined_chart_data = pd.DataFrame()
    if not df.empty:
        df['month_year'] = df['close_date'].dt.to_period('M').astype(str)
        combined_chart_data = (
            df.groupby(['month_year', 'is_forecast'])['total_revenue']
            .sum()
            .reset_index()
        )
        combined_chart_data['label'] = combined_chart_data['is_forecast'].apply(
            lambda x: 'Historical' if x == 0 else 'Forecast'
        )

    # ── Chart 2: Daily revenue bar chart ─────────────────────────────────────
    revenue_by_day = pd.DataFrame()
    if not df.empty:
        revenue_by_day = (
            df.groupby(['close_date', 'is_forecast'])['total_revenue']
            .sum()
            .reset_index()
        )
        revenue_by_day['label'] = revenue_by_day['is_forecast'].apply(
            lambda x: 'Historical' if x == 0 else 'Forecast'
        )

    # ── Chart 3: Top products bar chart ──────────────────────────────────────
    top_products = pd.DataFrame()
    if not df.empty:
        rev_per_product = (
            df.groupby(['product', 'is_forecast'])['total_revenue']
            .sum()
            .reset_index()
        )
        top_10 = (
            rev_per_product.groupby('product')['total_revenue']
            .sum()
            .nlargest(10)
            .index
        )
        top_products = rev_per_product[
            rev_per_product['product'].isin(top_10)
        ].copy()
        top_products['label'] = top_products['is_forecast'].apply(
            lambda x: 'Historical' if x == 0 else 'Forecast'
        )

    # ── Details table ─────────────────────────────────────────────────────────
    details = df.sort_values('close_date', ascending=False).copy()
    details['is_forecast'] = details['is_forecast'].map(
        {0: 'Historical', 1: 'Forecast'}
    )
    table_cols = ['close_date', 'product', 'series', 'office_location',
                  'type', 'total_revenue', 'total_deals', 'is_forecast']
    details = details[[c for c in table_cols if c in details.columns]]

    return (total_rev_str, total_deals_str, avg_deal_str,
            combined_chart_data, revenue_by_day, top_products, details)


# Dashboard Layout
st.title("CRM Revenue Forecast Dashboard")
st.markdown("Historical actuals (2021–2024) and machine learning forecast (2025) — "
            "powered by Random Forest.")
st.markdown("---")

# ── Filters row ───────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    selected_start = st.date_input(
        "Start Date",
        value=st.session_state.selected_start_date,
        key="selected_start_date"
    )
with col2:
    selected_end = st.date_input(
        "End Date",
        value=st.session_state.selected_end_date,
        key="selected_end_date"
    )
with col3:
    selected_product = st.selectbox(
        "Product",
        options=products_list,
        index=products_list.index(st.session_state.selected_product),
        key="selected_product"
    )
with col4:
    selected_location = st.selectbox(
        "Office Location",
        options=office_locations_list,
        index=office_locations_list.index(st.session_state.selected_office_location),
        key="selected_office_location"
    )
with col5:
    selected_deal_type = st.selectbox(
        "Deal Type",
        options=deal_types_list,
        index=deal_types_list.index(st.session_state.selected_deal_type),
        key="selected_deal_type"
    )

st.markdown("---")

# Apply filters and get updated KPIs, charts, and details table
(
    total_rev_str, total_deals_str, avg_deal_str,
    combined_chart_data, revenue_by_day, top_products, details
) = update_data(
    st.session_state.selected_start_date,
    st.session_state.selected_end_date,
    st.session_state.selected_product,
    st.session_state.selected_office_location,
    st.session_state.selected_deal_type
)

# ── KPI cards ─────────────────────────────────────────────────────────────────
kpi1, kpi2, kpi3 = st.columns(3)

with kpi1:
    st.markdown(f"""
        <div class="metric-card">
            <p class="metric-label">Total Revenue</p>
            <p class="metric-number">{total_rev_str}</p>
        </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown(f"""
        <div class="metric-card">
            <p class="metric-label">Total Deals Closed</p>
            <p class="metric-number metric-number-neutral">{total_deals_str}</p>
        </div>
    """, unsafe_allow_html=True)

with kpi3:
    st.markdown(f"""
        <div class="metric-card">
            <p class="metric-label">Avg Deal Size</p>
            <p class="metric-number">{avg_deal_str}</p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ── Chart selector & charts ───────────────────────────────────────────────────
selected_chart = st.radio("Select Chart", chart_list, horizontal=True)
st.markdown("")

if selected_chart == "Total Revenue Trend":
    st.subheader("Monthly Revenue — Historical vs Forecast")
    if not combined_chart_data.empty:
        chart = alt.Chart(combined_chart_data).mark_line(point=True).encode(
            x=alt.X('month_year:N',
                    sort=alt.EncodingSortField(field='month_year', op='min',
                                               order='ascending'),
                    title='Month'),
            y=alt.Y('total_revenue:Q', title='Total Revenue ($)'),
            color=alt.Color('label:N',
                            scale=alt.Scale(
                                domain=['Historical', 'Forecast'],
                                range=['#1976D2', '#FF6F00']
                            ),
                            title='Period'),
            strokeDash=alt.condition(
                alt.datum.label == 'Forecast',
                alt.value([5, 5]),   
                alt.value([0])      
            ),
            tooltip=[
                alt.Tooltip('month_year:N', title='Month'),
                alt.Tooltip('label:N',      title='Period'),
                alt.Tooltip('total_revenue:Q', format='$,.0f', title='Revenue')
            ]
        ).properties(height=400)
        st.altair_chart(chart, use_container_width=True)
    else:
        st.warning("No data available for the selected filters.")

elif selected_chart == "Revenue by Day":
    st.subheader("Daily Revenue — Historical vs Forecast")
    if not revenue_by_day.empty:
        chart = alt.Chart(revenue_by_day).mark_bar().encode(
            x=alt.X('close_date:T', title='Date'),
            y=alt.Y('total_revenue:Q', title='Total Revenue ($)'),
            color=alt.Color('label:N',
                            scale=alt.Scale(
                                domain=['Historical', 'Forecast'],
                                range=['#1976D2', '#FF6F00']
                            ),
                            title='Period'),
            tooltip=[
                alt.Tooltip('close_date:T',    title='Date'),
                alt.Tooltip('label:N',         title='Period'),
                alt.Tooltip('total_revenue:Q', format='$,.0f', title='Revenue')
            ]
        ).properties(height=400)
        st.altair_chart(chart, use_container_width=True)
    else:
        st.warning("No data available for the selected filters.")

elif selected_chart == "Top Products":
    st.subheader("Top 10 Products by Revenue — Historical vs Forecast")
    if not top_products.empty:
        chart = alt.Chart(top_products).mark_bar().encode(
            x=alt.X('total_revenue:Q', title='Total Revenue ($)'),
            y=alt.Y('product:N',
                    sort='-x',
                    title='Product'),
            color=alt.Color('label:N',
                            scale=alt.Scale(
                                domain=['Historical', 'Forecast'],
                                range=['#1976D2', '#FF6F00']
                            ),
                            title='Period'),
            tooltip=[
                alt.Tooltip('product:N',       title='Product'),
                alt.Tooltip('label:N',         title='Period'),
                alt.Tooltip('total_revenue:Q', format='$,.0f', title='Revenue')
            ]
        ).properties(height=400)
        st.altair_chart(chart, use_container_width=True)
    else:
        st.warning("No data available for the selected filters.")

st.markdown("---")

# ── Details table ─────────────────────────────────────────────────────────────
st.subheader("Transaction Detail")
st.caption("Sorted by most recent close date. Filters above apply to this table.")
st.dataframe(
    details,
    use_container_width=True,
    hide_index=True,
    column_config={
        "close_date":      st.column_config.DateColumn("Close Date"),
        "total_revenue":   st.column_config.NumberColumn("Revenue", format="$%.2f"),
        "total_deals":     st.column_config.NumberColumn("Deals"),
        "is_forecast":     st.column_config.TextColumn("Period"),
        "product":         st.column_config.TextColumn("Product"),
        "series":          st.column_config.TextColumn("Series"),
        "office_location": st.column_config.TextColumn("Location"),
        "type":            st.column_config.TextColumn("Deal Type"),
    }
)