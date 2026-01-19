import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import os
import subprocess

# Page configuration
st.set_page_config(
    page_title="NYC Motor Vehicle Collisions Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dataset file path (Hugging Face Spaces handles Git LFS automatically)
CSV_FILE = "NYC_crashes_dataset_STANDARDIZED.csv"

# Custom CSS for light theme styling
st.markdown("""
<style>
    /* Main header */
    .main-header {
        font-size: 2.2rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    /* Force light theme colors */
    .stApp {
        background-color: #ffffff !important;
        color: #333333 !important;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa !important;
        color: #333333 !important;
    }
    section[data-testid="stSidebar"] * {
        color: #333333 !important;
    }
    section[data-testid="stSidebar"] .stMarkdown {
        color: #333333 !important;
    }
    
    /* Main content text */
    .stMarkdown, .stText, p, span, label, h1, h2, h3, h4, h5, h6 {
        color: #333333 !important;
    }
    
    /* Reduce page title padding */
    .main-header {
        margin-top: 0 !important;
        margin-bottom: 0.5rem !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }
    
    /* Metric cards */
    .stMetric {
        background-color: #f8f9fa !important;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    [data-testid="stMetricValue"] {
        color: #333333 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #555555 !important;
    }
    
    /* Sidebar Quick Stats styling */
    [data-testid="stSidebar"] [data-testid="stMetricValue"] {
        font-size: 1rem !important;
        color: #333333 !important;
    }
    [data-testid="stSidebar"] [data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        color: #555555 !important;
    }
    
    /* Select boxes and inputs */
    .stSelectbox label, .stMultiSelect label, .stSlider label {
        color: #333333 !important;
    }
    .stSelectbox > div > div, .stMultiSelect > div > div {
        background-color: #ffffff !important;
        color: #333333 !important;
        border: 1px solid #cccccc !important;
    }
    
    /* Radio buttons */
    .stRadio label {
        color: #333333 !important;
    }
    .stRadio > div {
        background-color: transparent !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        color: #333333 !important;
        background-color: #f8f9fa !important;
    }
    .streamlit-expanderContent {
        color: #333333 !important;
        background-color: #ffffff !important;
    }
    
    /* Buttons */
    .stButton > button {
        color: #333333 !important;
        background-color: #f8f9fa !important;
        border: 1px solid #cccccc !important;
    }
    .stButton > button:hover {
        background-color: #e9ecef !important;
    }
    
    /* Info boxes */
    .stAlert {
        background-color: #e7f3ff !important;
        color: #333333 !important;
    }
    
    /* Subheaders */
    .stSubheader, [data-testid="stSubheader"] {
        color: #333333 !important;
    }
    
    /* Dataframe */
    .stDataFrame {
        background-color: #ffffff !important;
    }
    .stDataFrame td, .stDataFrame th {
        color: #333333 !important;
    }
    
    /* Caption text */
    .stCaption, [data-testid="stCaptionContainer"] {
        color: #666666 !important;
    }
    
    /* Slider */
    .stSlider > div > div > div {
        color: #333333 !important;
    }
    
    /* Multiselect tags */
    .stMultiSelect [data-baseweb="tag"] {
        background-color: #e9ecef !important;
        color: #333333 !important;
    }
    
    /* Dropdown menus */
    [data-baseweb="popover"] {
        background-color: #ffffff !important;
    }
    [data-baseweb="menu"] {
        background-color: #ffffff !important;
    }
    [data-baseweb="menu"] li {
        color: #333333 !important;
    }
    
    /* Column headers and dividers */
    hr {
        border-color: #cccccc !important;
    }
    
    /* Plotly chart container */
    .js-plotly-plot .plotly .modebar {
        background-color: transparent !important;
    }
    
    /* Tab styling if used */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #f8f9fa !important;
    }
    .stTabs [data-baseweb="tab"] {
        color: #333333 !important;
    }
    
    /* Checkbox */
    .stCheckbox label {
        color: #333333 !important;
    }
    
    /* Number input */
    .stNumberInput label {
        color: #333333 !important;
    }
    .stNumberInput input {
        color: #333333 !important;
        background-color: #ffffff !important;
    }
    
    /* Plot container with border for separation */
    .stPlotlyChart {
        background-color: #f8f9fa !important;
        border: 1px solid #e0e0e0 !important;
        border-radius: 8px !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Expander/Tooltip styling - light gray background */
    [data-testid="stExpander"] {
        background-color: #f0f2f6 !important;
        border: 1px solid #e0e0e0 !important;
        border-radius: 8px !important;
        margin-bottom: 1rem !important;
    }
    [data-testid="stExpander"] summary {
        color: #333333 !important;
        background-color: #f0f2f6 !important;
    }
    [data-testid="stExpander"] > div {
        background-color: #f8f9fa !important;
    }
    
    /* Fix select box dropdown - light gray */
    .stSelectbox [data-baseweb="select"] > div {
        background-color: #f8f9fa !important;
        color: #333333 !important;
    }
    .stMultiSelect [data-baseweb="select"] > div {
        background-color: #f8f9fa !important;
        color: #333333 !important;
    }
    
    /* Dropdown menu items */
    [data-baseweb="popover"] > div {
        background-color: #f8f9fa !important;
    }
    [data-baseweb="menu"] {
        background-color: #f8f9fa !important;
    }
    [data-baseweb="menu"] li {
        color: #333333 !important;
        background-color: #f8f9fa !important;
    }
    [data-baseweb="menu"] li:hover {
        background-color: #e9ecef !important;
    }
    
    /* Fix table/dataframe backgrounds */
    .stDataFrame, [data-testid="stDataFrame"] {
        background-color: #f8f9fa !important;
        border: 1px solid #e0e0e0 !important;
        border-radius: 8px !important;
    }
    .stDataFrame table {
        background-color: #f8f9fa !important;
    }
    .stDataFrame th {
        background-color: #e9ecef !important;
        color: #333333 !important;
    }
    .stDataFrame td {
        background-color: #f8f9fa !important;
        color: #333333 !important;
    }
    
    /* Input fields light gray background */
    input, textarea {
        background-color: #f8f9fa !important;
        color: #333333 !important;
        border: 1px solid #cccccc !important;
    }
    
    /* Subheader with top margin for separation */
    [data-testid="stSubheader"] {
        margin-top: 1.5rem !important;
        padding-top: 1rem !important;
        border-top: 1px solid #e0e0e0 !important;
    }
</style>
""", unsafe_allow_html=True)

def get_plot_layout():
    """Return consistent plot layout settings for light theme."""
    return {
        'template': 'plotly_white',
        'paper_bgcolor': '#ffffff',
        'plot_bgcolor': '#ffffff',
        'font': {'color': '#333333', 'size': 12},
        'title': {'font': {'color': '#333333', 'size': 16}},
        'xaxis': {'color': '#333333', 'tickfont': {'color': '#333333'}, 'title': {'font': {'color': '#333333'}}},
        'yaxis': {'color': '#333333', 'tickfont': {'color': '#333333'}, 'title': {'font': {'color': '#333333'}}},
        'legend': {'font': {'color': '#333333'}, 'orientation': 'h', 'yanchor': 'bottom', 'y': -0.3, 'xanchor': 'center', 'x': 0.5},
        'coloraxis': {'colorbar': {'tickfont': {'color': '#333333'}, 'title': {'font': {'color': '#333333'}}}}
    }

# Custom color scale: light blue (low) to dark blue (high)
BLUE_SCALE = [[0, '#cce5ff'], [0.25, '#99caff'], [0.5, '#4da3ff'], [0.75, '#1a75ff'], [1, '#0047b3']]
RED_SCALE = [[0, '#ffcccc'], [0.25, '#ff9999'], [0.5, '#ff6666'], [0.75, '#ff3333'], [1, '#cc0000']]

def add_percentage_to_bar(fig, data=None, value_col=None):
    """Add percentage labels to bar chart with proper contrast.
    
    Calculates total from all visible traces to handle grouped/stacked charts correctly.
    Arguments data and value_col are kept for compatibility but ignored in favor of figure data.
    """
    # Calculate grand total from all traces
    grand_total = 0
    for trace in fig.data:
        # Check if trace is a bar type (or compatible)
        if trace.type == 'bar':
            if trace.orientation == 'h':
                values = trace.x
            else:
                values = trace.y
            
            # Handle numpy arrays or tuples
            if values is not None:
                grand_total += sum([v for v in values if v is not None])
    
    # Update traces with percentage text
    for trace in fig.data:
        if trace.type == 'bar':
            if trace.orientation == 'h':
                values = trace.x
            else:
                values = trace.y
                
            if values is not None and grand_total > 0:
                # Calculate percentages
                percentages = [(v / grand_total) * 100 if v is not None else 0 for v in values]
                
                # Update text
                trace.text = [f'{p:.1f}%' for p in percentages]
                trace.textposition = 'outside'
                # Ensure text is visible (dark color, large font)
                trace.textfont = dict(size=14, color='#333333')
                # Prevent text from being clipped
                trace.cliponaxis = False
                
    return fig

def show_analysis(text, label="Analysis"):
    """Display context-aware analysis as a tooltip/expander."""
    with st.expander(f"ℹ️ {label}", expanded=False):
        st.write(text)

def show_page_subtitle(text):
    """Display page description as centered subtitle under main title."""
    st.markdown(f'<p style="text-align: center; color: #7f8c8d; font-size: 1.1em; margin-top: -10px; margin-bottom: 20px;">{text}</p>', unsafe_allow_html=True)

def create_sortable_bar_chart(data, x_col, y_col, title, key_prefix, color=None, color_scale=None, orientation='v', height=500):
    """Create a bar chart with sorting options."""
    sort_col1, sort_col2 = st.columns([3, 1])
    
    with sort_col1:
        sort_option = st.selectbox(
            "Sort by:",
            ["Default", f"{y_col} (High to Low)", f"{y_col} (Low to High)", f"{x_col} (A-Z)", f"{x_col} (Z-A)"],
            key=f"{key_prefix}_sort"
        )
    
    with sort_col2:
        if st.button("Reset", key=f"{key_prefix}_reset"):
            sort_option = "Default"
    
    # Apply sorting
    sorted_data = data.copy()
    if sort_option == f"{y_col} (High to Low)":
        sorted_data = sorted_data.sort_values(y_col, ascending=False)
    elif sort_option == f"{y_col} (Low to High)":
        sorted_data = sorted_data.sort_values(y_col, ascending=True)
    elif sort_option == f"{x_col} (A-Z)":
        sorted_data = sorted_data.sort_values(x_col, ascending=True)
    elif sort_option == f"{x_col} (Z-A)":
        sorted_data = sorted_data.sort_values(x_col, ascending=False)
    
    # Create chart
    if orientation == 'h':
        if color_scale:
            fig = px.bar(sorted_data, x=y_col, y=x_col, orientation='h', title=title,
                        color=y_col, color_continuous_scale=color_scale)
        else:
            fig = px.bar(sorted_data, x=y_col, y=x_col, orientation='h', title=title,
                        color=color if color else None)
        if sort_option == "Default":
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    else:
        if color_scale:
            fig = px.bar(sorted_data, x=x_col, y=y_col, title=title,
                        color=y_col, color_continuous_scale=color_scale)
        else:
            fig = px.bar(sorted_data, x=x_col, y=y_col, title=title,
                        color=color if color else None)
    
    fig.update_layout(height=height, **get_plot_layout())
    return fig, sorted_data

# Initialize session state for interactive filtering
if 'selected_year' not in st.session_state:
    st.session_state.selected_year = None
if 'selected_borough' not in st.session_state:
    st.session_state.selected_borough = None
if 'selected_hour' not in st.session_state:
    st.session_state.selected_hour = None
if 'selected_factor' not in st.session_state:
    st.session_state.selected_factor = None

# Data loading with caching
@st.cache_data(show_spinner="Loading NYC Crashes data...")
def load_data():
    """Load and preprocess the NYC crashes dataset with standardized vehicle types."""
    # Use the CSV_FILE constant defined at the top
    df = pd.read_csv(
        CSV_FILE,
        parse_dates=['CRASH DATE'],
        low_memory=False
    )
    
    # Data is already cleaned and standardized
    # - Invalid coordinates removed
    # - Missing values filled
    # - Vehicle types standardized (Taxi/TAXI/taxi → Taxi, Unknown/UNK/Unkown → Unknown)
    
    # Parse time
    df['CRASH TIME'] = pd.to_datetime(df['CRASH TIME'], format='%H:%M', errors='coerce')
    df['HOUR'] = df['CRASH TIME'].dt.hour
    
    # Extract date components
    df['YEAR'] = df['CRASH DATE'].dt.year
    df['MONTH'] = df['CRASH DATE'].dt.month
    df['DAY_OF_WEEK'] = df['CRASH DATE'].dt.dayofweek
    df['DAY_NAME'] = df['CRASH DATE'].dt.day_name()
    df['MONTH_NAME'] = df['CRASH DATE'].dt.month_name()
    
    # Total casualties
    df['TOTAL_INJURED'] = df['NUMBER OF PERSONS INJURED'].fillna(0)
    df['TOTAL_KILLED'] = df['NUMBER OF PERSONS KILLED'].fillna(0)
    
    # Clean borough data - rename UNKNOWN to Highways (crashes on highways/bridges without specific borough)
    df['BOROUGH'] = df['BOROUGH'].fillna('Highways')
    df['BOROUGH'] = df['BOROUGH'].replace('UNKNOWN', 'Highways')
    
    return df

@st.cache_data
def get_aggregated_data(df):
    """Pre-aggregate data for faster visualizations."""
    agg_data = {}
    
    # Yearly trends
    agg_data['yearly'] = df.groupby('YEAR').agg({
        'COLLISION_ID': 'count',
        'TOTAL_INJURED': 'sum',
        'TOTAL_KILLED': 'sum',
        'NUMBER OF PEDESTRIANS INJURED': 'sum',
        'NUMBER OF PEDESTRIANS KILLED': 'sum',
        'NUMBER OF CYCLIST INJURED': 'sum',
        'NUMBER OF CYCLIST KILLED': 'sum',
        'NUMBER OF MOTORIST INJURED': 'sum',
        'NUMBER OF MOTORIST KILLED': 'sum'
    }).reset_index()
    agg_data['yearly'].columns = ['Year', 'Crashes', 'Injured', 'Killed', 
                                   'Pedestrians Injured', 'Pedestrians Killed',
                                   'Cyclists Injured', 'Cyclists Killed',
                                   'Motorists Injured', 'Motorists Killed']
    
    # Monthly trends
    agg_data['monthly'] = df.groupby(['YEAR', 'MONTH']).agg({
        'COLLISION_ID': 'count',
        'TOTAL_INJURED': 'sum',
        'TOTAL_KILLED': 'sum'
    }).reset_index()
    agg_data['monthly']['DATE'] = pd.to_datetime(agg_data['monthly'][['YEAR', 'MONTH']].assign(DAY=1))
    
    # By borough
    agg_data['borough'] = df.groupby('BOROUGH').agg({
        'COLLISION_ID': 'count',
        'TOTAL_INJURED': 'sum',
        'TOTAL_KILLED': 'sum'
    }).reset_index()
    agg_data['borough'].columns = ['Borough', 'Crashes', 'Injured', 'Killed']
    
    # By hour
    agg_data['hourly'] = df.groupby('HOUR').agg({
        'COLLISION_ID': 'count',
        'TOTAL_INJURED': 'sum',
        'TOTAL_KILLED': 'sum'
    }).reset_index()
    agg_data['hourly'].columns = ['Hour', 'Crashes', 'Injured', 'Killed']
    
    # By day of week
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    agg_data['daily'] = df.groupby(['DAY_OF_WEEK', 'DAY_NAME']).agg({
        'COLLISION_ID': 'count',
        'TOTAL_INJURED': 'sum',
        'TOTAL_KILLED': 'sum'
    }).reset_index()
    agg_data['daily'].columns = ['Day_Num', 'Day', 'Crashes', 'Injured', 'Killed']
    agg_data['daily'] = agg_data['daily'].sort_values('Day_Num')
    
    # Contributing factors
    factors = df['CONTRIBUTING FACTOR VEHICLE 1'].value_counts().head(15).reset_index()
    factors.columns = ['Factor', 'Count']
    agg_data['factors'] = factors
    
    # Vehicle types
    vehicles = df['VEHICLE TYPE CODE 1'].value_counts().head(15).reset_index()
    vehicles.columns = ['Vehicle Type', 'Count']
    agg_data['vehicles'] = vehicles
    
    # Heatmap data (hour vs day)
    agg_data['heatmap'] = df.groupby(['DAY_OF_WEEK', 'HOUR']).size().reset_index(name='Crashes')
    
    return agg_data

@st.cache_data
def get_map_sample(df, sample_size=50000):
    """Get a sample of data for mapping (data already has valid coordinates)."""
    if len(df) > sample_size:
        return df.sample(n=sample_size, random_state=42)
    return df

# Load data
df = load_data()
agg_data = get_aggregated_data(df)

# Sidebar
st.sidebar.markdown("## NYC Crashes Dashboard")
st.sidebar.markdown("---")

# Navigation
page = st.sidebar.radio(
    "Navigate to:",
    ["Overview", "Geographic Analysis", "Temporal Analysis", 
     "Cause Analysis", "Severity Analysis", "Risk Prediction"]
)

# Year filter
years = sorted(df['YEAR'].dropna().unique())
year_range = st.sidebar.slider(
    "Select Year Range:",
    min_value=int(min(years)),
    max_value=int(max(years)),
    value=(2013, int(max(years)))  # Default to 2013-2025 (2012 has partial data)
)

# Borough filter
boroughs = ['All'] + sorted([b for b in df['BOROUGH'].unique() if b != 'Highways'])
selected_borough = st.sidebar.selectbox("Select Borough:", boroughs)

# Filter data based on selections
@st.cache_data
def filter_data(df, year_range, borough):
    filtered = df[(df['YEAR'] >= year_range[0]) & (df['YEAR'] <= year_range[1])]
    if borough != 'All':
        filtered = filtered[filtered['BOROUGH'] == borough]
    return filtered

filtered_df = filter_data(df, year_range, selected_borough)


# ============== PAGE: OVERVIEW ==============
if page == "Overview":
    st.markdown('<h1 class="main-header">NYC Motor Vehicle Collisions Dashboard</h1>', unsafe_allow_html=True)
    show_page_subtitle("Comprehensive analysis of motor vehicle collisions in New York City. Explore crash trends, patterns, and safety insights across all five boroughs.")
    
    # Get available years from filtered data
    all_years = sorted(filtered_df['YEAR'].unique().tolist())
    
    # Initialize filter variables with defaults
    selected_years = all_years
    severity_filter = ["Fatal", "Injury", "No Injury"]
    month_filter = list(range(1, 13))
    
    # Apply page-specific filters (use defaults initially, will be updated by expander)
    page_df = filtered_df.copy()
    
    # KPI Metrics first
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_crashes = len(page_df)
    total_injured = int(page_df['TOTAL_INJURED'].sum())
    total_killed = int(page_df['TOTAL_KILLED'].sum())
    pedestrians_killed = int(page_df['NUMBER OF PEDESTRIANS KILLED'].sum())
    cyclists_killed = int(page_df['NUMBER OF CYCLIST KILLED'].sum())
    
    col1.metric("Total Crashes", f"{total_crashes:,}")
    col2.metric("Total Injured", f"{total_injured:,}")
    col3.metric("Total Killed", f"{total_killed:,}")
    col4.metric("Pedestrians Killed", f"{pedestrians_killed:,}")
    col5.metric("Cyclists Killed", f"{cyclists_killed:,}")
    
    # Page-specific filters (now below stats)
    with st.expander("Additional Filters", expanded=False):
        filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
        with filter_col1:
            # Year filter for cross-filtering
            selected_years = st.multiselect(
                "Filter by Year:",
                all_years,
                default=all_years,
                key="overview_year_filter"
            )
        with filter_col2:
            severity_filter = st.multiselect(
                "Severity:",
                ["Fatal", "Injury", "No Injury"],
                default=["Fatal", "Injury", "No Injury"],
                key="overview_severity"
            )
        with filter_col3:
            month_filter = st.multiselect(
                "Months:",
                list(range(1, 13)),
                default=list(range(1, 13)),
                format_func=lambda x: ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][x-1],
                key="overview_month"
            )
        with filter_col4:
            if st.button("Reset Filters", key="overview_reset_filters"):
                st.rerun()
    
    # Re-apply page-specific filters after expander
    page_df = filtered_df.copy()
    if severity_filter:
        severity_conditions = []
        if "Fatal" in severity_filter:
            severity_conditions.append(page_df['TOTAL_KILLED'] > 0)
        if "Injury" in severity_filter:
            severity_conditions.append((page_df['TOTAL_INJURED'] > 0) & (page_df['TOTAL_KILLED'] == 0))
        if "No Injury" in severity_filter:
            severity_conditions.append((page_df['TOTAL_INJURED'] == 0) & (page_df['TOTAL_KILLED'] == 0))
        if severity_conditions:
            combined = severity_conditions[0]
            for cond in severity_conditions[1:]:
                combined = combined | cond
            page_df = page_df[combined]
    if month_filter:
        page_df = page_df[page_df['MONTH'].isin(month_filter)]
    
    # Overview analysis (now below filters)
    avg_daily = len(page_df) / max((page_df['CRASH DATE'].max() - page_df['CRASH DATE'].min()).days, 1) if len(page_df) > 0 else 0
    fatality_rate = (page_df['TOTAL_KILLED'].sum() / len(page_df)) * 100 if len(page_df) > 0 else 0
    show_analysis(f"This dashboard analyzes {len(page_df):,} motor vehicle collisions in New York City. On average, there are approximately {avg_daily:.0f} crashes per day in the selected period, with a fatality rate of {fatality_rate:.3f}% per crash.", "Dashboard Overview")
    
    st.markdown("---")
    
    # Apply year filter (now in Additional Filters expander)
    if selected_years and len(selected_years) < len(all_years):
        cross_filtered_df = page_df[page_df['YEAR'].isin(selected_years)]
    else:
        cross_filtered_df = page_df
    
    # Prepare yearly data for visualization
    yearly_filtered = cross_filtered_df.groupby('YEAR').agg({
        'COLLISION_ID': 'count',
        'TOTAL_INJURED': 'sum',
        'TOTAL_KILLED': 'sum'
    }).reset_index()
    yearly_filtered.columns = ['Year', 'Crashes', 'Injured', 'Killed']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Crashes Over Time")
        fig = px.line(yearly_filtered, x='Year', y='Crashes', 
                      markers=True, title='Annual Crash Trend')
        if selected_years and len(selected_years) < len(all_years):
            for yr in selected_years:
                fig.add_vline(x=yr, line_dash="dash", line_color="red", opacity=0.5)
        fig.update_layout(height=700, **get_plot_layout())
        st.plotly_chart(fig, width='stretch')
        
        if len(yearly_filtered) > 1:
            trend = "increasing" if yearly_filtered['Crashes'].iloc[-1] > yearly_filtered['Crashes'].iloc[0] else "decreasing"
            peak_year = yearly_filtered.loc[yearly_filtered['Crashes'].idxmax(), 'Year']
            show_analysis(f"The annual crash trend shows a {trend} pattern over the selected period. The peak year was {int(peak_year)} with {yearly_filtered['Crashes'].max():,} crashes.", "Trend Insight")
    
    with col2:
        st.subheader("Crashes by Borough")
        borough_filtered = cross_filtered_df.groupby('BOROUGH').size().reset_index(name='Crashes')
        borough_filtered = borough_filtered[borough_filtered['BOROUGH'] != 'Highways']
        
        fig = px.pie(borough_filtered, values='Crashes', names='BOROUGH',
                     title='Distribution by Borough', hole=0.4)
        fig.update_layout(height=700, **get_plot_layout())
        st.plotly_chart(fig, width='stretch')
        
        if len(borough_filtered) > 0:
            top_borough = borough_filtered.loc[borough_filtered['Crashes'].idxmax()]
            pct = (top_borough['Crashes'] / borough_filtered['Crashes'].sum()) * 100
            show_analysis(f"{top_borough['BOROUGH']} has the highest number of crashes, accounting for {pct:.1f}% of all collisions in the selected period.", "Borough Insight")
    
    # Monthly trend
    st.subheader("Monthly Crash Trend")
    monthly_filtered = cross_filtered_df.groupby([cross_filtered_df['CRASH DATE'].dt.to_period('M')]).size().reset_index(name='Crashes')
    monthly_filtered['Date'] = monthly_filtered['CRASH DATE'].dt.to_timestamp()
    
    fig = px.area(monthly_filtered, x='Date', y='Crashes', title='Monthly Crashes')
    fig.update_layout(height=700, **get_plot_layout())
    st.plotly_chart(fig, width='stretch')
    
    show_analysis("The monthly trend reveals seasonal patterns in crash frequency. Variations may be influenced by weather conditions, holiday periods, and traffic volume changes throughout the year.", "Seasonal Patterns")
    
    # Casualties breakdown with sorting
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Injuries by Category")
        injury_data = pd.DataFrame({
            'Category': ['Pedestrians', 'Cyclists', 'Motorists'],
            'Injured': [
                cross_filtered_df['NUMBER OF PEDESTRIANS INJURED'].sum(),
                cross_filtered_df['NUMBER OF CYCLIST INJURED'].sum(),
                cross_filtered_df['NUMBER OF MOTORIST INJURED'].sum()
            ]
        })
        
        sort_inj = st.selectbox("Sort:", ["Default", "High to Low", "Low to High"], key="injury_sort")
        if sort_inj == "High to Low":
            injury_data = injury_data.sort_values('Injured', ascending=False)
        elif sort_inj == "Low to High":
            injury_data = injury_data.sort_values('Injured', ascending=True)
        
        fig = px.bar(injury_data, x='Category', y='Injured',
                     title='Injuries by Road User Type',
                     color='Injured', color_continuous_scale=BLUE_SCALE)
        fig = add_percentage_to_bar(fig, injury_data, 'Injured')
        fig.update_layout(height=700, showlegend=False, **get_plot_layout())
        st.plotly_chart(fig, width='stretch')
        
        total_injuries = injury_data['Injured'].sum()
        if total_injuries > 0:
            motorist_pct = (injury_data[injury_data['Category'] == 'Motorists']['Injured'].values[0] / total_injuries) * 100
            show_analysis(f"Motorists account for {motorist_pct:.1f}% of all injuries, while pedestrians and cyclists represent vulnerable road users requiring targeted safety interventions.", "Injury Breakdown")
    
    with col2:
        st.subheader("Fatalities by Category")
        fatality_data = pd.DataFrame({
            'Category': ['Pedestrians', 'Cyclists', 'Motorists'],
            'Killed': [
                cross_filtered_df['NUMBER OF PEDESTRIANS KILLED'].sum(),
                cross_filtered_df['NUMBER OF CYCLIST KILLED'].sum(),
                cross_filtered_df['NUMBER OF MOTORIST KILLED'].sum()
            ]
        })
        
        sort_fat = st.selectbox("Sort:", ["Default", "High to Low", "Low to High"], key="fatality_sort")
        if sort_fat == "High to Low":
            fatality_data = fatality_data.sort_values('Killed', ascending=False)
        elif sort_fat == "Low to High":
            fatality_data = fatality_data.sort_values('Killed', ascending=True)
        
        fig = px.bar(fatality_data, x='Category', y='Killed',
                     title='Fatalities by Road User Type',
                     color='Killed', color_continuous_scale=RED_SCALE)
        fig = add_percentage_to_bar(fig, fatality_data, 'Killed')
        fig.update_layout(height=700, showlegend=False, **get_plot_layout())
        st.plotly_chart(fig, width='stretch')
        
        total_fatalities = fatality_data['Killed'].sum()
        if total_fatalities > 0:
            ped_pct = (fatality_data[fatality_data['Category'] == 'Pedestrians']['Killed'].values[0] / total_fatalities) * 100
            show_analysis(f"Pedestrians represent {ped_pct:.1f}% of all fatalities, highlighting the critical need for pedestrian safety measures and infrastructure improvements.", "Fatality Insight")

# ============== PAGE: GEOGRAPHIC ANALYSIS ==============
elif page == "Geographic Analysis":
    st.markdown('<h1 class="main-header">Geographic Analysis</h1>', unsafe_allow_html=True)
    show_page_subtitle("Visualize the spatial distribution of crashes across NYC. Identify hotspots and high-risk areas for targeted safety interventions.")
    st.markdown("---")
    
    # Page-specific filters
    with st.expander("Additional Filters", expanded=False):
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        with filter_col1:
            geo_severity = st.multiselect(
                "Severity:",
                ["Fatal", "Injury", "No Injury"],
                default=["Fatal", "Injury", "No Injury"],
                key="geo_severity"
            )
        with filter_col2:
            hour_range = st.slider("Hour Range:", 0, 23, (0, 23), key="geo_hour")
        with filter_col3:
            if st.button("Reset Filters", key="geo_reset"):
                st.rerun()
    
    # Apply filters
    page_df = filtered_df.copy()
    if geo_severity:
        severity_conditions = []
        if "Fatal" in geo_severity:
            severity_conditions.append(page_df['TOTAL_KILLED'] > 0)
        if "Injury" in geo_severity:
            severity_conditions.append((page_df['TOTAL_INJURED'] > 0) & (page_df['TOTAL_KILLED'] == 0))
        if "No Injury" in geo_severity:
            severity_conditions.append((page_df['TOTAL_INJURED'] == 0) & (page_df['TOTAL_KILLED'] == 0))
        if severity_conditions:
            combined = severity_conditions[0]
            for cond in severity_conditions[1:]:
                combined = combined | cond
            page_df = page_df[combined]
    page_df = page_df[(page_df['HOUR'] >= hour_range[0]) & (page_df['HOUR'] <= hour_range[1])]
    
    
    # Get map data
    map_df = get_map_sample(page_df, sample_size=50000)
    
    st.info(f"Displaying {len(map_df):,} crashes with valid coordinates (sampled for performance)")
    
    # Map type selection
    map_type = st.radio("Select Map Type:", ["Scatter Map", "Density Heatmap", "Cluster Map"], horizontal=True)
    
    if map_type == "Scatter Map":
        # Color by severity
        color_col, _ = st.columns([1, 3])
        with color_col:
            color_by = st.selectbox("Color by:", ["Borough", "Severity", "Hour"])
        
        if color_by == "Borough":
            fig = px.scatter_map(
                map_df, lat='LATITUDE', lon='LONGITUDE',
                color='BOROUGH', zoom=10,
                map_style='carto-positron',
                title='Crash Locations by Borough',
                opacity=0.5
            )
        elif color_by == "Severity":
            map_df['Severity'] = map_df.apply(
                lambda x: 'Fatal' if x['TOTAL_KILLED'] > 0 else ('Injury' if x['TOTAL_INJURED'] > 0 else 'No Injury'),
                axis=1
            )
            fig = px.scatter_map(
                map_df, lat='LATITUDE', lon='LONGITUDE',
                color='Severity', zoom=10,
                map_style='carto-positron',
                title='Crash Locations by Severity',
                color_discrete_map={'Fatal': 'red', 'Injury': 'orange', 'No Injury': 'green'},
                opacity=0.6
            )
        else:
            fig = px.scatter_map(
                map_df, lat='LATITUDE', lon='LONGITUDE',
                color='HOUR', zoom=10,
                map_style='carto-positron',
                title='Crash Locations by Hour',
                color_continuous_scale='Viridis',
                opacity=0.5
            )
        
        layout = get_plot_layout()
        layout['legend'] = dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.01)
        layout['margin'] = dict(t=40, b=0, l=0, r=0)
        fig.update_layout(height=700, **layout)
        st.plotly_chart(fig, width='stretch', config={'modeBarButtonsToRemove': ['select2d', 'lasso2d']})
        
        if color_by == "Borough":
            show_analysis("Each color represents a different borough. This view helps identify which areas of the city experience the most crashes and reveals geographic patterns in collision distribution.", "Map Legend")
        elif color_by == "Severity":
            show_analysis("Red markers indicate fatal crashes, orange indicates injury crashes, and green shows property-damage-only incidents. This helps prioritize areas with the most severe outcomes.", "Severity Legend")
        else:
            show_analysis("The color gradient shows the time of day when crashes occurred. This can reveal patterns related to commute times, nightlife areas, and visibility conditions.", "Time Legend")
    
    elif map_type == "Density Heatmap":
        fig = px.density_map(
            map_df, lat='LATITUDE', lon='LONGITUDE',
            radius=5, zoom=10,
            map_style='carto-positron',
            title='Crash Density Heatmap',
            color_continuous_scale='Oranges'
        )
        layout = get_plot_layout()
        layout['legend'] = dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.01)
        layout['margin'] = dict(t=40, b=0, l=0, r=0)
        fig.update_layout(height=700, **layout)
        st.plotly_chart(fig, width='stretch', config={'modeBarButtonsToRemove': ['select2d', 'lasso2d']})
        
        show_analysis("The heatmap intensity shows crash concentration. Darker orange areas indicate higher crash density/severity, helping identify dangerous corridors and intersections that may require infrastructure improvements.", "Heatmap Guide")
    
    else:  # Cluster Map - Borough Highlights
        st.subheader("Crash Distribution by Borough")
        
        # Borough-level aggregation
        borough_stats = filtered_df[filtered_df['BOROUGH'] != 'Highways'].groupby('BOROUGH').agg({
            'COLLISION_ID': 'count',
            'TOTAL_INJURED': 'sum',
            'TOTAL_KILLED': 'sum'
        }).reset_index()
        borough_stats.columns = ['Borough', 'Crashes', 'Injured', 'Killed']
        
        # Get crash points for each borough to show distribution
        borough_map_df = map_df[map_df['BOROUGH'] != 'Highways'].copy()
        
        # Merge borough crash counts for gradient intensity
        borough_map_df = borough_map_df.merge(
            borough_stats[['Borough', 'Crashes']], 
            left_on='BOROUGH', 
            right_on='Borough', 
            how='left'
        )
        
        # Create scatter map with borough colors and gradient intensity based on crash count
        fig = px.scatter_map(
            borough_map_df, 
            lat='LATITUDE', 
            lon='LONGITUDE',
            color='BOROUGH',
            size='Crashes',
            hover_data=['BOROUGH', 'Crashes'],
            zoom=10,
            map_style='carto-positron',
            title='Crash Distribution by Borough (Size = Total Borough Crashes)',
            opacity=0.5,
            size_max=15,
            color_discrete_map={
                'MANHATTAN': '#d62728',
                'BROOKLYN': '#ff7f0e',
                'QUEENS': '#2ca02c',
                'BRONX': '#9467bd',
                'STATEN ISLAND': '#1f77b4'
            }
        )
        layout = get_plot_layout()
        layout['legend'] = dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.01, title="Borough")
        layout['margin'] = dict(t=40, b=0, l=0, r=0)
        fig.update_layout(height=700, **layout)
        st.plotly_chart(fig, width='stretch', config={'modeBarButtonsToRemove': ['select2d', 'lasso2d']})
        
        # Show borough statistics table
        st.markdown("**Borough Statistics:**")
        borough_stats_display = borough_stats.sort_values('Crashes', ascending=False)
        borough_stats_display['Crashes'] = borough_stats_display['Crashes'].apply(lambda x: f"{x:,}")
        borough_stats_display['Injured'] = borough_stats_display['Injured'].apply(lambda x: f"{int(x):,}")
        borough_stats_display['Killed'] = borough_stats_display['Killed'].apply(lambda x: f"{int(x):,}")
        st.dataframe(borough_stats_display, hide_index=True, width='stretch')
        
        show_analysis("Each color represents a different borough, showing the geographic distribution of crashes. This view helps identify which areas within each borough have higher crash concentrations.", "Borough Distribution Guide")
    
    # Top dangerous streets
    st.markdown("---")
    st.subheader("Most Dangerous Streets")
    
    col1, col2 = st.columns(2)
    
    with col1:
        top_streets = page_df['ON STREET NAME'].value_counts().head(15).reset_index()
        top_streets.columns = ['Street', 'Crashes']
        
        sort_streets = st.selectbox("Sort:", ["Crashes (High to Low)", "Crashes (Low to High)", "Street (A-Z)"], key="streets_sort")
        if sort_streets == "Crashes (High to Low)":
            top_streets = top_streets.sort_values('Crashes', ascending=True)
        elif sort_streets == "Crashes (Low to High)":
            top_streets = top_streets.sort_values('Crashes', ascending=False)
        elif sort_streets == "Street (A-Z)":
            top_streets = top_streets.sort_values('Street', ascending=True)
        
        fig = px.bar(top_streets, x='Crashes', y='Street', orientation='h',
                     title='Top 15 Streets by Crash Count',
                     color='Crashes', color_continuous_scale=BLUE_SCALE)
        fig = add_percentage_to_bar(fig, top_streets, 'Crashes')
        fig.update_layout(height=700, **get_plot_layout())
        st.plotly_chart(fig, width='stretch')
        
        if len(top_streets) > 0:
            # Find the street with highest crashes from original data
            highest_street = page_df['ON STREET NAME'].value_counts().head(1)
            show_analysis(f"{highest_street.index[0]} has the highest crash count with {highest_street.values[0]:,} incidents. Major arterial roads and highways typically appear at the top due to higher traffic volumes.", "Street Insight")
    
    with col2:
        # Streets with most fatalities
        street_fatalities = page_df.groupby('ON STREET NAME')['TOTAL_KILLED'].sum().sort_values(ascending=False).head(15).reset_index()
        street_fatalities.columns = ['Street', 'Fatalities']
        
        sort_fatal_streets = st.selectbox("Sort:", ["Fatalities (High to Low)", "Fatalities (Low to High)", "Street (A-Z)"], key="fatal_streets_sort")
        if sort_fatal_streets == "Fatalities (High to Low)":
            street_fatalities = street_fatalities.sort_values('Fatalities', ascending=True)
        elif sort_fatal_streets == "Fatalities (Low to High)":
            street_fatalities = street_fatalities.sort_values('Fatalities', ascending=False)
        elif sort_fatal_streets == "Street (A-Z)":
            street_fatalities = street_fatalities.sort_values('Street', ascending=True)
        
        fig = px.bar(street_fatalities, x='Fatalities', y='Street', orientation='h',
                     title='Top 15 Streets by Fatalities',
                     color='Fatalities', color_continuous_scale=RED_SCALE)
        fig = add_percentage_to_bar(fig, street_fatalities, 'Fatalities')
        fig.update_layout(height=700, **get_plot_layout())
        st.plotly_chart(fig, width='stretch')
        
        if len(street_fatalities) > 0:
            show_analysis("Streets with the most fatalities may differ from those with the most crashes. High-speed roads and areas with vulnerable road users often have disproportionately high fatality rates.", "Fatality Insight")

# ============== PAGE: TEMPORAL ANALYSIS ==============
elif page == "Temporal Analysis":
    st.markdown('<h1 class="main-header">Temporal Analysis</h1>', unsafe_allow_html=True)
    show_page_subtitle("Understand when crashes occur. Identify high-risk time periods related to rush hours, weekends, and seasonal variations.")
    st.markdown("---")
    
    # Page-specific filters
    with st.expander("Additional Filters", expanded=False):
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        with filter_col1:
            temp_severity = st.multiselect(
                "Severity:",
                ["Fatal", "Injury", "No Injury"],
                default=["Fatal", "Injury", "No Injury"],
                key="temp_severity"
            )
        with filter_col2:
            day_filter = st.multiselect(
                "Days:",
                ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                default=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                key="temp_days"
            )
        with filter_col3:
            if st.button("Reset Filters", key="temp_reset"):
                st.rerun()
    
    # Apply filters
    page_df = filtered_df.copy()
    if temp_severity:
        severity_conditions = []
        if "Fatal" in temp_severity:
            severity_conditions.append(page_df['TOTAL_KILLED'] > 0)
        if "Injury" in temp_severity:
            severity_conditions.append((page_df['TOTAL_INJURED'] > 0) & (page_df['TOTAL_KILLED'] == 0))
        if "No Injury" in temp_severity:
            severity_conditions.append((page_df['TOTAL_INJURED'] == 0) & (page_df['TOTAL_KILLED'] == 0))
        if severity_conditions:
            combined = severity_conditions[0]
            for cond in severity_conditions[1:]:
                combined = combined | cond
            page_df = page_df[combined]
    if day_filter:
        page_df = page_df[page_df['DAY_NAME'].isin(day_filter)]
    
    
    # Cross-filter by hour
    selected_hour = st.selectbox("Filter all charts by Hour:", ["All Hours"] + list(range(24)), key="temp_hour_filter")
    if selected_hour != "All Hours":
        cross_df = page_df[page_df['HOUR'] == selected_hour]
    else:
        cross_df = page_df
    
    # Hourly pattern
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Crashes by Hour of Day")
        hourly = page_df.groupby('HOUR').size().reset_index(name='Crashes')
        
        sort_hourly = st.selectbox("Sort:", ["Hour (0-23)", "Crashes (High to Low)", "Crashes (Low to High)"], key="hourly_sort")
        if sort_hourly == "Crashes (High to Low)":
            hourly = hourly.sort_values('Crashes', ascending=False)
        elif sort_hourly == "Crashes (Low to High)":
            hourly = hourly.sort_values('Crashes', ascending=True)
        
        fig = px.bar(hourly, x='HOUR', y='Crashes', 
                     title='Hourly Distribution of Crashes',
                     labels={'HOUR': 'Hour of Day'},
                     color='Crashes', color_continuous_scale=BLUE_SCALE)
        fig = add_percentage_to_bar(fig, hourly, 'Crashes')
        if selected_hour != "All Hours":
            fig.add_vline(x=selected_hour, line_dash="dash", line_color="red")
        fig.update_layout(height=700, **get_plot_layout())
        fig.update_xaxes(tickmode='linear', dtick=2)
        st.plotly_chart(fig, width='stretch')
        
        if len(hourly) > 0:
            peak_hour = hourly.loc[hourly['Crashes'].idxmax()]
            show_analysis(f"Peak crash hour is {int(peak_hour['HOUR'])}:00 with {peak_hour['Crashes']:,} crashes. Afternoon hours typically see higher crash rates due to increased traffic and driver fatigue.", "Peak Hour Insight")
    
    with col2:
        st.subheader("Crashes by Day of Week")
        daily = cross_df.groupby(['DAY_OF_WEEK', 'DAY_NAME']).size().reset_index(name='Crashes')
        daily = daily.sort_values('DAY_OF_WEEK')
        
        sort_daily = st.selectbox("Sort:", ["Day Order", "Crashes (High to Low)", "Crashes (Low to High)"], key="daily_sort")
        if sort_daily == "Crashes (High to Low)":
            daily = daily.sort_values('Crashes', ascending=False)
        elif sort_daily == "Crashes (Low to High)":
            daily = daily.sort_values('Crashes', ascending=True)
        
        fig = px.bar(daily, x='DAY_NAME', y='Crashes',
                     title='Daily Distribution of Crashes',
                     color='Crashes', color_continuous_scale=BLUE_SCALE)
        fig = add_percentage_to_bar(fig, daily, 'Crashes')
        fig.update_layout(height=700, **get_plot_layout())
        st.plotly_chart(fig, width='stretch')
        
        if len(daily) > 0:
            peak_day = daily.loc[daily['Crashes'].idxmax()]
            show_analysis(f"{peak_day['DAY_NAME']} has the highest crash count. Weekday patterns often differ from weekends due to commuter traffic versus recreational driving.", "Peak Day Insight")
    
    # Heatmap: Hour vs Day
    st.subheader("Crash Heatmap: Hour vs Day of Week")
    heatmap_data = page_df.groupby(['DAY_OF_WEEK', 'HOUR']).size().reset_index(name='Crashes')
    heatmap_pivot = heatmap_data.pivot(index='DAY_OF_WEEK', columns='HOUR', values='Crashes').fillna(0)
    
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    fig = px.imshow(
        heatmap_pivot.values,
        x=[f"{h}:00" for h in range(24)],
        y=day_names,
        color_continuous_scale='YlOrRd',
        title='Crash Frequency by Hour and Day',
        labels={'color': 'Crashes'}
    )
    fig.update_layout(height=700, **get_plot_layout())
    st.plotly_chart(fig, width='stretch')
    
    show_analysis("The heatmap reveals when crashes are most concentrated. Darker colors indicate higher crash frequency. Notice patterns during weekday rush hours versus weekend late-night periods.", "Heatmap Guide")
    
    # Monthly pattern
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Crashes by Month")
        monthly = cross_df.groupby('MONTH').size().reset_index(name='Crashes')
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly['Month_Name'] = monthly['MONTH'].apply(lambda x: month_names[int(x)-1] if pd.notna(x) else 'Unknown')
        fig = px.line(monthly, x='Month_Name', y='Crashes', markers=True,
                      title='Monthly Pattern')
        fig.update_layout(height=700, **get_plot_layout())
        st.plotly_chart(fig, width='stretch')
        
        show_analysis("Monthly patterns often reflect seasonal factors. Winter months may show fewer crashes due to reduced travel, while summer months may see increases from tourism and construction.", "Monthly Patterns")
    
    with col2:
        st.subheader("Year-over-Year Comparison")
        yoy = cross_df.groupby(['YEAR', 'MONTH']).size().reset_index(name='Crashes')
        fig = px.line(yoy, x='MONTH', y='Crashes', color='YEAR',
                      title='Monthly Crashes by Year',
                      labels={'MONTH': 'Month'})
        fig.update_layout(height=700, **get_plot_layout())
        fig.update_xaxes(tickmode='linear', dtick=1)
        st.plotly_chart(fig, width='stretch')
        
        show_analysis("Comparing years reveals long-term trends and the impact of policy changes, infrastructure improvements, or external events like the COVID-19 pandemic on crash rates.", "Year Comparison")
    
    # Rush hour analysis
    st.markdown("---")
    st.subheader("Rush Hour Analysis")
    
    page_df['Period'] = page_df['HOUR'].apply(
        lambda x: 'Morning Rush (7-9)' if 7 <= x <= 9 
        else ('Evening Rush (4-7)' if 16 <= x <= 19 
        else ('Night (10pm-6am)' if x >= 22 or x <= 6 
        else 'Midday (10am-4pm)'))
    )
    
    period_stats = page_df.groupby('Period').agg({
        'COLLISION_ID': 'count',
        'TOTAL_INJURED': 'sum',
        'TOTAL_KILLED': 'sum'
    }).reset_index()
    period_stats.columns = ['Period', 'Crashes', 'Injured', 'Killed']
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.pie(period_stats, values='Crashes', names='Period',
                     title='Crashes by Time Period', hole=0.3)
        fig.update_layout(height=700, **get_plot_layout())
        st.plotly_chart(fig, width='stretch')
        
        if len(period_stats) > 0:
            top_period = period_stats.loc[period_stats['Crashes'].idxmax()]
            show_analysis(f"The {top_period['Period']} period accounts for the most crashes. Understanding these patterns helps allocate traffic enforcement and emergency response resources effectively.", "Rush Hour Insight")
    
    with col2:
        fig = px.bar(period_stats, x='Period', y=['Injured', 'Killed'],
                     title='Casualties by Time Period', barmode='group')
        fig = add_percentage_to_bar(fig)
        fig.update_layout(height=700, **get_plot_layout())
        st.plotly_chart(fig, width='stretch')
        
        show_analysis("Night-time crashes often result in more severe outcomes due to reduced visibility, higher speeds, and impaired driving, even though total crash counts may be lower.", "Severity by Time")

# ============== PAGE: CAUSE ANALYSIS ==============
elif page == "Cause Analysis":
    st.markdown('<h1 class="main-header">Cause & Vehicle Analysis</h1>', unsafe_allow_html=True)
    show_page_subtitle("Examine the primary factors contributing to crashes. Identify common causes and vehicle types involved in collisions.")
    st.markdown("---")
    
    # Page-specific filters
    with st.expander("Additional Filters", expanded=False):
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        with filter_col1:
            cause_severity = st.multiselect(
                "Severity:",
                ["Fatal", "Injury", "No Injury"],
                default=["Fatal", "Injury", "No Injury"],
                key="cause_severity"
            )
        with filter_col2:
            top_n = st.slider("Number of items to show:", 5, 20, 15, key="cause_top_n")
        with filter_col3:
            if st.button("Reset Filters", key="cause_reset"):
                st.rerun()
    
    # Apply filters
    page_df = filtered_df.copy()
    if cause_severity:
        severity_conditions = []
        if "Fatal" in cause_severity:
            severity_conditions.append(page_df['TOTAL_KILLED'] > 0)
        if "Injury" in cause_severity:
            severity_conditions.append((page_df['TOTAL_INJURED'] > 0) & (page_df['TOTAL_KILLED'] == 0))
        if "No Injury" in cause_severity:
            severity_conditions.append((page_df['TOTAL_INJURED'] == 0) & (page_df['TOTAL_KILLED'] == 0))
        if severity_conditions:
            combined = severity_conditions[0]
            for cond in severity_conditions[1:]:
                combined = combined | cond
            page_df = page_df[combined]
    
    show_analysis("This section examines the primary causes of crashes and the types of vehicles involved. Understanding these factors is essential for developing targeted prevention strategies and policy interventions.", "About Cause Analysis")
    
    # Cross-filter by contributing factor
    all_factors = page_df['CONTRIBUTING FACTOR VEHICLE 1'].value_counts().head(20).index.tolist()
    all_factors = [f for f in all_factors if f != 'Unspecified']
    selected_factor = st.selectbox("Filter all charts by Contributing Factor:", ["All Factors"] + all_factors, key="cause_factor_filter")
    
    if selected_factor != "All Factors":
        cross_df = page_df[page_df['CONTRIBUTING FACTOR VEHICLE 1'] == selected_factor]
    else:
        cross_df = page_df
    
    # Contributing factors
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top Contributing Factors")
        factors = cross_df['CONTRIBUTING FACTOR VEHICLE 1'].value_counts().head(top_n).reset_index()
        factors.columns = ['Factor', 'Count']
        factors = factors[factors['Factor'] != 'Unspecified']
        
        # Sorting options
        sort_factors = st.selectbox("Sort:", ["Count (High to Low)", "Count (Low to High)", "Factor (A-Z)", "Factor (Z-A)"], key="factors_sort")
        if sort_factors == "Count (High to Low)":
            factors = factors.sort_values('Count', ascending=True)
        elif sort_factors == "Count (Low to High)":
            factors = factors.sort_values('Count', ascending=False)
        elif sort_factors == "Factor (A-Z)":
            factors = factors.sort_values('Factor', ascending=False)
        else:
            factors = factors.sort_values('Factor', ascending=True)
        
        fig = px.bar(factors, x='Count', y='Factor', orientation='h',
                     title=f'Top {top_n} Contributing Factors',
                     color='Count', color_continuous_scale=BLUE_SCALE)
        fig = add_percentage_to_bar(fig, factors, 'Count')
        fig.update_layout(height=700, **get_plot_layout())
        st.plotly_chart(fig, width='stretch')
        
        # Get the actual top factor from the original data (before sorting for display)
        top_factor_actual = cross_df['CONTRIBUTING FACTOR VEHICLE 1'].value_counts()
        top_factor_actual = top_factor_actual[top_factor_actual.index != 'Unspecified']
        if len(top_factor_actual) > 0:
            top_factor_name = top_factor_actual.index[0]
            top_factor_count = top_factor_actual.values[0]
            show_analysis(f"{top_factor_name} is the leading contributing factor with {top_factor_count:,} incidents. Driver behavior factors dominate the list, suggesting education and enforcement as key intervention strategies.", "Top Factor Insight")
    
    with col2:
        st.subheader("Vehicle Types Involved")
        vehicles = cross_df['VEHICLE TYPE CODE 1'].value_counts().head(top_n).reset_index()
        vehicles.columns = ['Vehicle', 'Count']
        
        # Sorting options
        sort_vehicles = st.selectbox("Sort:", ["Count (High to Low)", "Count (Low to High)", "Vehicle (A-Z)", "Vehicle (Z-A)"], key="vehicles_sort")
        if sort_vehicles == "Count (High to Low)":
            vehicles = vehicles.sort_values('Count', ascending=True)
        elif sort_vehicles == "Count (Low to High)":
            vehicles = vehicles.sort_values('Count', ascending=False)
        elif sort_vehicles == "Vehicle (A-Z)":
            vehicles = vehicles.sort_values('Vehicle', ascending=False)
        else:
            vehicles = vehicles.sort_values('Vehicle', ascending=True)
        
        fig = px.bar(vehicles, x='Count', y='Vehicle', orientation='h',
                     title=f'Top {top_n} Vehicle Types',
                     color='Count', color_continuous_scale=BLUE_SCALE)
        fig = add_percentage_to_bar(fig, vehicles, 'Count')
        fig.update_layout(height=700, **get_plot_layout())
        st.plotly_chart(fig, width='stretch')
        
        if len(vehicles) > 0:
            show_analysis("Sedans and SUVs dominate crash statistics due to their prevalence on NYC roads. Commercial vehicles and taxis also appear frequently due to their high mileage in urban environments.", "Vehicle Insight")
    
    # Factor analysis by severity
    st.markdown("---")
    st.subheader("Contributing Factors in Fatal Crashes")
    
    fatal_crashes = cross_df[cross_df['TOTAL_KILLED'] > 0]
    fatal_factors = fatal_crashes['CONTRIBUTING FACTOR VEHICLE 1'].value_counts().head(10).reset_index()
    fatal_factors.columns = ['Factor', 'Fatal Crashes']
    fatal_factors = fatal_factors[fatal_factors['Factor'] != 'Unspecified']
    
    # Sorting for fatal factors
    sort_fatal = st.selectbox("Sort:", ["Fatal Crashes (High to Low)", "Fatal Crashes (Low to High)", "Factor (A-Z)"], key="fatal_factors_sort")
    if sort_fatal == "Fatal Crashes (Low to High)":
        fatal_factors = fatal_factors.sort_values('Fatal Crashes', ascending=True)
    elif sort_fatal == "Factor (A-Z)":
        fatal_factors = fatal_factors.sort_values('Factor', ascending=True)
    
    fig = px.bar(fatal_factors, x='Factor', y='Fatal Crashes',
                 title='Top Contributing Factors in Fatal Crashes',
                 color='Fatal Crashes', color_continuous_scale=RED_SCALE)
    fig = add_percentage_to_bar(fig, fatal_factors, 'Fatal Crashes')
    fig.update_layout(height=700, **get_plot_layout())
    fig.update_xaxes(tickangle=45)
    # Move colorbar to the right side of the chart
    fig.update_layout(
        coloraxis_colorbar=dict(orientation='v', x=1.02, y=0.5, yanchor='middle', len=0.8),
        margin=dict(r=120)
    )
    st.plotly_chart(fig, width='stretch')
    
    show_analysis("Fatal crashes often involve different contributing factors than non-fatal crashes. Unsafe speed, failure to yield, and driver inattention are particularly dangerous behaviors that increase fatality risk.", "Fatal Crash Factors")
    
    # Factor trends over time
    st.subheader("Contributing Factor Trends Over Time")
    
    top_factors = filtered_df['CONTRIBUTING FACTOR VEHICLE 1'].value_counts().head(5).index.tolist()
    if 'Unspecified' in top_factors:
        top_factors.remove('Unspecified')
        next_factor = filtered_df['CONTRIBUTING FACTOR VEHICLE 1'].value_counts().iloc[5:6].index.tolist()
        top_factors.extend(next_factor)
    
    factor_trends = filtered_df[filtered_df['CONTRIBUTING FACTOR VEHICLE 1'].isin(top_factors)]
    factor_yearly = factor_trends.groupby(['YEAR', 'CONTRIBUTING FACTOR VEHICLE 1']).size().reset_index(name='Crashes')
    
    fig = px.line(factor_yearly, x='YEAR', y='Crashes', 
                  color='CONTRIBUTING FACTOR VEHICLE 1',
                  title='Trend of Top Contributing Factors',
                  markers=True)
    fig.update_layout(height=700, legend_title='Factor', **get_plot_layout())
    st.plotly_chart(fig, width='stretch')
    
    show_analysis("Tracking contributing factors over time reveals the effectiveness of safety campaigns and policy changes. Decreasing trends may indicate successful interventions, while increasing trends highlight emerging concerns.", "Trend Analysis")
    
    # Relationship: Vehicle Type vs Contributing Factor
    st.markdown("---")
    st.subheader("Vehicle Type vs. Contributing Factor")
    
    # Get top vehicles and factors
    top_vehicles_rel = page_df['VEHICLE TYPE CODE 1'].value_counts().head(10).index.tolist()
    top_factors_rel = page_df['CONTRIBUTING FACTOR VEHICLE 1'].value_counts().head(10).index.tolist()
    if 'Unspecified' in top_factors_rel:
        top_factors_rel.remove('Unspecified')
    
    rel_df = page_df[
        (page_df['VEHICLE TYPE CODE 1'].isin(top_vehicles_rel)) & 
        (page_df['CONTRIBUTING FACTOR VEHICLE 1'].isin(top_factors_rel))
    ]
    
    rel_counts = rel_df.groupby(['VEHICLE TYPE CODE 1', 'CONTRIBUTING FACTOR VEHICLE 1']).size().reset_index(name='Count')
    rel_pivot = rel_counts.pivot(index='CONTRIBUTING FACTOR VEHICLE 1', columns='VEHICLE TYPE CODE 1', values='Count').fillna(0)
    
    fig = px.imshow(
        rel_pivot,
        labels=dict(x="Vehicle Type", y="Contributing Factor", color="Crashes"),
        x=rel_pivot.columns,
        y=rel_pivot.index,
        color_continuous_scale='Blues',
        title='Relationship Matrix: Vehicle Types & Contributing Factors'
    )
    fig.update_layout(height=700, **get_plot_layout())
    st.plotly_chart(fig, width='stretch')
    
    show_analysis("This heatmap highlights specific risks associated with different vehicle types. Darker squares indicate frequent associations between a vehicle type and a contributing factor.", "Relationship Insight")

    # Vehicle Specific Analysis
    st.subheader("Deep Dive: Contributing Factors by Vehicle Type")
    
    vehicle_list = page_df['VEHICLE TYPE CODE 1'].value_counts().head(20).index.tolist()
    selected_vehicle_drill = st.selectbox("Select Vehicle Type to Analyze:", vehicle_list, key="vehicle_drill")
    
    vehicle_specific_df = page_df[page_df['VEHICLE TYPE CODE 1'] == selected_vehicle_drill]
    vehicle_factors = vehicle_specific_df['CONTRIBUTING FACTOR VEHICLE 1'].value_counts().head(10).reset_index()
    vehicle_factors.columns = ['Factor', 'Count']
    vehicle_factors = vehicle_factors[vehicle_factors['Factor'] != 'Unspecified']
    # Sort ascending so highest value appears at top in horizontal bar chart
    vehicle_factors = vehicle_factors.sort_values('Count', ascending=True)
    
    fig = px.bar(vehicle_factors, x='Count', y='Factor', orientation='h',
                 title=f'Top Contributing Factors for {selected_vehicle_drill}',
                 color='Count', color_continuous_scale=BLUE_SCALE)
    fig = add_percentage_to_bar(fig, vehicle_factors, 'Count')
    fig.update_layout(height=700, **get_plot_layout())
    st.plotly_chart(fig, width='stretch')
    
    show_analysis(f"Analyzing {selected_vehicle_drill} specifically helps target interventions. Distinct vehicle types often exhibit unique crash causalities based on their size, usage patterns, and blind spots.", "Vehicle Specific Insight")
    
    # Vehicle type by borough
    st.subheader("Vehicle Types by Borough")
    
    vehicle_borough = filtered_df[filtered_df['BOROUGH'] != 'Highways'].groupby(
        ['BOROUGH', 'VEHICLE TYPE CODE 1']
    ).size().reset_index(name='Crashes')
    
    top_vehicles = filtered_df['VEHICLE TYPE CODE 1'].value_counts().head(5).index.tolist()
    vehicle_borough = vehicle_borough[vehicle_borough['VEHICLE TYPE CODE 1'].isin(top_vehicles)]
    
    fig = px.bar(vehicle_borough, x='BOROUGH', y='Crashes', 
                 color='VEHICLE TYPE CODE 1',
                 title='Vehicle Type Distribution by Borough',
                 barmode='group')
    fig = add_percentage_to_bar(fig)
    fig.update_layout(height=700, legend_title='Vehicle Type', **get_plot_layout())
    st.plotly_chart(fig, width='stretch')
    
    show_analysis("Vehicle type distribution varies by borough based on land use patterns. Manhattan sees more taxis and commercial vehicles, while outer boroughs have higher proportions of personal vehicles.", "Borough Patterns")

# ============== PAGE: SEVERITY ANALYSIS ==============
elif page == "Severity Analysis":
    st.markdown('<h1 class="main-header">Severity Analysis</h1>', unsafe_allow_html=True)
    show_page_subtitle("Analyze crash outcomes and severity. Understand the relationship between crash characteristics and injury/fatality rates.")
    st.markdown("---")
    
    # Page-specific filters
    with st.expander("Additional Filters", expanded=False):
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        with filter_col1:
            sev_year = st.multiselect(
                "Years:",
                sorted(filtered_df['YEAR'].dropna().unique().tolist()),
                default=sorted(filtered_df['YEAR'].dropna().unique().tolist()),
                key="sev_years"
            )
        with filter_col2:
            sev_borough = st.multiselect(
                "Boroughs:",
                [b for b in filtered_df['BOROUGH'].unique() if b != 'Highways'],
                default=[b for b in filtered_df['BOROUGH'].unique() if b != 'Highways'],
                key="sev_boroughs"
            )
        with filter_col3:
            if st.button("Reset Filters", key="sev_reset"):
                st.rerun()
    
    # Apply filters
    page_df = filtered_df.copy()
    if sev_year:
        page_df = page_df[page_df['YEAR'].isin(sev_year)]
    if sev_borough:
        page_df = page_df[page_df['BOROUGH'].isin(sev_borough)]
    
    show_analysis("This section focuses on crash outcomes, distinguishing between fatal, injury, and property-damage-only crashes. Understanding severity patterns helps prioritize safety investments for maximum life-saving impact.", "About Severity Analysis")
    
    # Severity KPIs
    fatal_crashes = page_df[page_df['TOTAL_KILLED'] > 0]
    injury_crashes = page_df[(page_df['TOTAL_INJURED'] > 0) & (page_df['TOTAL_KILLED'] == 0)]
    no_injury = page_df[(page_df['TOTAL_INJURED'] == 0) & (page_df['TOTAL_KILLED'] == 0)]
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Fatal Crashes", f"{len(fatal_crashes):,}")
    col2.metric("Injury Crashes", f"{len(injury_crashes):,}")
    col3.metric("No Injury Crashes", f"{len(no_injury):,}")
    col4.metric("Fatality Rate", f"{len(fatal_crashes)/len(page_df)*100:.2f}%" if len(page_df) > 0 else "0%")
    
    st.markdown("---")
    
    # Severity distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Crash Severity Distribution")
        severity_data = pd.DataFrame({
            'Severity': ['Fatal', 'Injury Only', 'No Injury'],
            'Count': [len(fatal_crashes), len(injury_crashes), len(no_injury)]
        })
        fig = px.pie(severity_data, values='Count', names='Severity',
                     color='Severity',
                     color_discrete_map={'Fatal': '#e74c3c', 'Injury Only': '#f39c12', 'No Injury': '#27ae60'},
                     title='Crash Severity Distribution')
        fig.update_layout(height=700, **get_plot_layout())
        st.plotly_chart(fig, width='stretch')
        
        no_injury_pct = (len(no_injury) / len(page_df)) * 100 if len(page_df) > 0 else 0
        show_analysis(f"The majority ({no_injury_pct:.1f}%) of crashes result in no injuries. However, even minor crashes have significant economic costs and traffic disruption impacts.", "Severity Breakdown")
    
    with col2:
        st.subheader("Fatal Crashes Over Time")
        fatal_yearly = fatal_crashes.groupby('YEAR').size().reset_index(name='Fatal Crashes')
        fig = px.line(fatal_yearly, x='YEAR', y='Fatal Crashes', markers=True,
                      title='Annual Fatal Crashes Trend')
        fig.update_layout(height=700, **get_plot_layout())
        fig.update_traces(line_color='#e74c3c')
        st.plotly_chart(fig, width='stretch')
        
        if len(fatal_yearly) > 1:
            trend = "decreasing" if fatal_yearly['Fatal Crashes'].iloc[-1] < fatal_yearly['Fatal Crashes'].iloc[0] else "increasing"
            show_analysis(f"The trend in fatal crashes is {trend} over the selected period. Vision Zero initiatives aim to eliminate all traffic fatalities through engineering, enforcement, and education.", "Fatal Trend")
    
    # Fatalities by victim type
    st.subheader("Fatalities by Victim Type Over Time")
    
    victim_yearly = page_df.groupby('YEAR').agg({
        'NUMBER OF PEDESTRIANS KILLED': 'sum',
        'NUMBER OF CYCLIST KILLED': 'sum',
        'NUMBER OF MOTORIST KILLED': 'sum'
    }).reset_index()
    victim_yearly.columns = ['Year', 'Pedestrians', 'Cyclists', 'Motorists']
    
    fig = px.line(victim_yearly, x='Year', y=['Pedestrians', 'Cyclists', 'Motorists'],
                  title='Fatalities by Victim Type Over Time', markers=True)
    fig.update_layout(height=700, legend_title='Victim Type', **get_plot_layout())
    st.plotly_chart(fig, width='stretch')
    
    show_analysis("Pedestrians and cyclists are particularly vulnerable road users. Trends in these categories often reflect the success of protected infrastructure investments like bike lanes and pedestrian plazas.", "Vulnerable Users")
    
    # Borough severity comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Fatalities by Borough")
        borough_fatal = page_df[page_df['BOROUGH'] != 'Highways'].groupby('BOROUGH').agg({
            'TOTAL_KILLED': 'sum'
        }).reset_index()
        borough_fatal.columns = ['Borough', 'Fatalities']
        
        sort_boro = st.selectbox("Sort:", ["Fatalities (High to Low)", "Fatalities (Low to High)", "Borough (A-Z)"], key="sev_boro_sort")
        if sort_boro == "Fatalities (Low to High)":
            borough_fatal = borough_fatal.sort_values('Fatalities', ascending=True)
        elif sort_boro == "Borough (A-Z)":
            borough_fatal = borough_fatal.sort_values('Borough', ascending=True)
        else:
            borough_fatal = borough_fatal.sort_values('Fatalities', ascending=False)
        
        fig = px.bar(borough_fatal, 
                     x='Fatalities', y='Borough', orientation='h',
                     title='Total Fatalities by Borough',
                     color='Fatalities', color_continuous_scale=RED_SCALE)
        fig = add_percentage_to_bar(fig, borough_fatal, 'Fatalities')
        fig.update_layout(height=700, **get_plot_layout())
        st.plotly_chart(fig, width='stretch')
        
        if len(borough_fatal) > 0:
            top_fatal_borough = borough_fatal.loc[borough_fatal['Fatalities'].idxmax()]
            show_analysis(f"{top_fatal_borough['Borough']} has the highest fatality count. Fatality rates should be considered alongside population and traffic volume for accurate risk assessment.", "Borough Fatalities")
    
    with col2:
        st.subheader("Fatal Crashes by Hour")
        fatal_hourly = fatal_crashes.groupby('HOUR').size().reset_index(name='Fatal Crashes')
        
        sort_hour = st.selectbox("Sort:", ["Hour (0-23)", "Fatal Crashes (High to Low)", "Fatal Crashes (Low to High)"], key="sev_hour_sort")
        if sort_hour == "Fatal Crashes (High to Low)":
            fatal_hourly = fatal_hourly.sort_values('Fatal Crashes', ascending=False)
        elif sort_hour == "Fatal Crashes (Low to High)":
            fatal_hourly = fatal_hourly.sort_values('Fatal Crashes', ascending=True)
        
        fig = px.bar(fatal_hourly, x='HOUR', y='Fatal Crashes',
                     title='Fatal Crashes by Hour of Day',
                     color='Fatal Crashes', color_continuous_scale=RED_SCALE)
        fig = add_percentage_to_bar(fig, fatal_hourly, 'Fatal Crashes')
        fig.update_layout(height=700, **get_plot_layout())
        fig.update_xaxes(tickmode='linear', dtick=2)
        st.plotly_chart(fig, width='stretch')
        
        if len(fatal_hourly) > 0:
            peak_fatal_hour = fatal_hourly.loc[fatal_hourly['Fatal Crashes'].idxmax()]
            show_analysis(f"Fatal crashes peak around {int(peak_fatal_hour['HOUR'])}:00. Late night and early morning hours often have higher fatality rates due to reduced visibility and impaired driving.", "Peak Fatal Hour")
    
    # Scatter plot with distributions: Injuries vs Hour colored by Borough
    st.markdown("---")
    st.subheader("Crash Severity Distribution Analysis")
    
    # Aggregate data by hour and borough for scatter plot
    scatter_data = page_df.groupby(['HOUR', 'BOROUGH']).agg({
        'TOTAL_INJURED': 'sum',
        'TOTAL_KILLED': 'sum',
        'COLLISION_ID': 'count'
    }).reset_index()
    scatter_data.columns = ['Hour', 'Borough', 'Total Injured', 'Total Killed', 'Crash Count']
    scatter_data = scatter_data[scatter_data['Borough'] != 'Highways']
    
    # Create scatter plot with trendline
    fig = px.scatter(
        scatter_data,
        x='Total Injured',
        y='Crash Count',
        color='Borough',
        hover_data=['Hour', 'Total Killed'],
        title='Injuries vs Crash Count by Borough',
        trendline='ols',
        opacity=0.7
    )
    # Make trendlines dotted linear lines
    for trace in fig.data:
        if trace.mode == 'lines':
            trace.line.dash = 'dot'
    fig.update_layout(height=700, **get_plot_layout())
    fig.update_layout(
        legend=dict(orientation="v", yanchor="top", y=0.95, xanchor="left", x=1.02)
    )
    st.plotly_chart(fig, width='stretch')
    
    show_analysis("This scatter plot reveals the relationship between injuries and crash frequency across boroughs. The trendlines show the linear correlation for each borough - steeper lines indicate stronger relationships between injury counts and crash frequency.", "Scatter Analysis")
    
    # Second scatter: Hour vs Injuries with severity coloring
    col1, col2 = st.columns(2)
    
    with col1:
        # Hourly severity scatter
        hourly_severity = page_df.groupby('HOUR').agg({
            'TOTAL_INJURED': 'sum',
            'TOTAL_KILLED': 'sum',
            'COLLISION_ID': 'count'
        }).reset_index()
        hourly_severity.columns = ['Hour', 'Injuries', 'Fatalities', 'Crashes']
        hourly_severity['Injury Rate'] = (hourly_severity['Injuries'] / hourly_severity['Crashes'] * 100).round(2)
        
        fig = px.scatter(
            hourly_severity,
            x='Hour',
            y='Injuries',
            size='Crashes',
            color='Fatalities',
            color_continuous_scale=RED_SCALE,
            title='Hourly Injury Pattern (size = crashes, color = fatalities)',
            trendline='lowess',
            trendline_options=dict(frac=0.3)
        )
        # Make trendline dotted
        for trace in fig.data:
            if trace.mode == 'lines':
                trace.line.dash = 'dot'
        fig.update_layout(height=500, **get_plot_layout())
        st.plotly_chart(fig, width='stretch')
        
        show_analysis("Each point represents an hour of the day. The curved trendline shows the non-linear pattern between hour and injuries. Peak injury hours often align with rush hour traffic.", "Hourly Pattern")
    
    with col2:
        # Day of week severity scatter
        daily_severity = page_df.groupby(['DAY_OF_WEEK', 'DAY_NAME']).agg({
            'TOTAL_INJURED': 'sum',
            'TOTAL_KILLED': 'sum',
            'COLLISION_ID': 'count'
        }).reset_index()
        daily_severity.columns = ['Day_Num', 'Day', 'Injuries', 'Fatalities', 'Crashes']
        daily_severity = daily_severity.sort_values('Day_Num')
        
        fig = px.scatter(
            daily_severity,
            x='Day_Num',
            y='Injuries',
            size='Crashes',
            color='Fatalities',
            color_continuous_scale=RED_SCALE,
            title='Daily Injury Pattern (size = crashes, color = fatalities)',
            trendline='lowess',
            trendline_options=dict(frac=0.5),
            labels={'Day_Num': 'Day of Week'}
        )
        # Make trendline dotted
        for trace in fig.data:
            if trace.mode == 'lines':
                trace.line.dash = 'dot'
        # Update x-axis to show day names instead of numbers
        fig.update_xaxes(
            tickmode='array',
            tickvals=[0, 1, 2, 3, 4, 5, 6],
            ticktext=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        )
        fig.update_layout(height=500, **get_plot_layout())
        st.plotly_chart(fig, width='stretch')
        
        show_analysis("Weekend days often show different patterns than weekdays. The curved trendline shows the overall weekly pattern. Severity can be higher on weekends due to recreational driving and nightlife activities.", "Daily Pattern")

    # Multi-fatality crashes
    st.markdown("---")
    st.subheader("Multi-Fatality Crashes")
    
    multi_fatal = page_df[page_df['TOTAL_KILLED'] > 1]
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.metric("Multi-Fatality Crashes", f"{len(multi_fatal):,}")
        st.metric("Total Deaths in Multi-Fatal", f"{int(multi_fatal['TOTAL_KILLED'].sum()):,}")
        
        multi_fatal_dist = multi_fatal['TOTAL_KILLED'].value_counts().sort_index().reset_index()
        multi_fatal_dist.columns = ['Deaths per Crash', 'Count']
        st.dataframe(multi_fatal_dist, hide_index=True)
    
    with col2:
        if len(multi_fatal) > 0:
            multi_fatal_factors = multi_fatal['CONTRIBUTING FACTOR VEHICLE 1'].value_counts().head(10).reset_index()
            multi_fatal_factors.columns = ['Factor', 'Count']
            multi_fatal_factors = multi_fatal_factors[multi_fatal_factors['Factor'] != 'Unspecified']
            
            fig = px.bar(multi_fatal_factors, x='Factor', y='Count',
                         title='Contributing Factors in Multi-Fatality Crashes',
                         color='Count', color_continuous_scale=RED_SCALE)
            fig = add_percentage_to_bar(fig, multi_fatal_factors, 'Count')
            fig.update_layout(height=500, margin=dict(b=100), **get_plot_layout())
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, width='stretch')
            
            show_analysis("Multi-fatality crashes are rare but devastating events. They often involve high speeds, impaired driving, or commercial vehicles, and warrant special investigative attention.", "Multi-Fatal Insight")

# ============== PAGE: RISK PREDICTION ==============
elif page == "Risk Prediction":
    st.markdown('<h1 class="main-header">Risk Prediction (Monte Carlo Simulation)</h1>', unsafe_allow_html=True)
    show_page_subtitle("Predict crash risk for specific locations using Monte Carlo simulation. Run thousands of iterations based on historical patterns to estimate future crash probabilities.")
    
    # Initialize filter variables with defaults
    risk_borough = 'All'
    min_crashes = 10
    selected_street = None
    
    # Get default street list for initial stats display
    street_counts_default = filtered_df['ON STREET NAME'].value_counts()
    street_counts_default = street_counts_default[street_counts_default.index.notna()]
    street_counts_default = street_counts_default[street_counts_default >= 10]
    default_street = street_counts_default.index[0] if len(street_counts_default) > 0 else None
    
    # Historical statistics first (like Overview page)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Streets", f"{filtered_df['ON STREET NAME'].nunique():,}")
    col2.metric("Total Crashes", f"{len(filtered_df):,}")
    col3.metric("Total Injuries", f"{int(filtered_df['TOTAL_INJURED'].sum()):,}")
    col4.metric("Total Fatalities", f"{int(filtered_df['TOTAL_KILLED'].sum()):,}")
    
    # About section first
    show_analysis("This page uses Monte Carlo simulation to predict crash risk for specific locations. The simulation runs thousands of iterations based on historical patterns to estimate future crash probabilities with confidence intervals.", "About Risk Prediction")
    
    # Location selection (now below About section)
    with st.expander("Select Location for Risk Analysis", expanded=True):
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            # Borough filter first to narrow down streets
            risk_borough = st.selectbox("Filter by Borough:", ['All'] + sorted([b for b in filtered_df['BOROUGH'].unique() if b != 'Highways']), key="risk_borough")
        
        with col2:
            # Minimum crash threshold filter
            min_crashes = st.selectbox("Min. Historical Crashes:", [1, 5, 10, 25, 50, 100, 250, 500, 1000], index=2, key="min_crashes",
                                       help="Filter to streets with at least this many historical crashes for more reliable predictions")
        
        # Get all streets, sorted by crash count, filtered by borough if selected
        if risk_borough != 'All':
            borough_df = filtered_df[filtered_df['BOROUGH'] == risk_borough]
        else:
            borough_df = filtered_df
        
        # Get all streets with at least min_crashes, sorted by frequency
        street_counts = borough_df['ON STREET NAME'].value_counts()
        street_counts = street_counts[street_counts.index.notna()]
        street_counts = street_counts[street_counts >= min_crashes]
        all_streets = street_counts.index.tolist()
        
        with col3:
            if len(all_streets) > 0:
                selected_street = st.selectbox(f"Select Street ({len(all_streets):,} available):", all_streets, key="risk_street")
            else:
                st.warning("No streets match criteria.")
                selected_street = None
    
    # Only proceed if a street is selected
    if selected_street is None:
        st.stop()
    
    # Filter data for selected location - use full dataset for stats, filtered for charts
    location_df_full = df[df['ON STREET NAME'] == selected_street]
    if risk_borough != 'All':
        location_df_full = location_df_full[location_df_full['BOROUGH'] == risk_borough]
    
    # For Historical Crash Trend chart, respect sidebar year filter
    location_df = filtered_df[filtered_df['ON STREET NAME'] == selected_street]
    if risk_borough != 'All':
        location_df = location_df[location_df['BOROUGH'] == risk_borough]
    
    # Location-specific statistics
    st.markdown("---")
    st.subheader(f"Historical Statistics for {selected_street}")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Crashes", f"{len(location_df):,}")
    col2.metric("Total Injuries", f"{int(location_df['TOTAL_INJURED'].sum()):,}")
    col3.metric("Total Fatalities", f"{int(location_df['TOTAL_KILLED'].sum()):,}")
    
    # Calculate average crashes per year
    years_in_data = location_df['YEAR'].nunique()
    avg_per_year = len(location_df) / max(years_in_data, 1)
    col4.metric("Avg Crashes/Year", f"{avg_per_year:.1f}")
    
    # Monte Carlo Simulation
    st.markdown("---")
    st.subheader("Monte Carlo Crash Risk Simulation")
    
    col1, col2 = st.columns(2)
    with col1:
        n_simulations = st.slider("Number of Simulations:", 1000, 10000, 5000, step=1000, key="n_sims")
    with col2:
        forecast_months = st.slider("Forecast Period (months):", 1, 24, 12, key="forecast_months")
    
    if st.button("Run Monte Carlo Simulation", type="primary", key="run_mc"):
        with st.spinner("Running simulation..."):
            # Calculate monthly crash rate from full historical data (not filtered by year)
            monthly_data = location_df_full.groupby([location_df_full['CRASH DATE'].dt.to_period('M')]).size()
            
            if len(monthly_data) > 0:
                mean_monthly = monthly_data.mean()
                std_monthly = monthly_data.std() if len(monthly_data) > 1 else mean_monthly * 0.3
                
                # Run Monte Carlo simulation using Poisson distribution (appropriate for count data)
                np.random.seed(42)
                
                # Simulate monthly crashes for the forecast period
                simulated_totals = []
                for _ in range(n_simulations):
                    # Use Poisson distribution for crash counts
                    monthly_crashes = np.random.poisson(lam=max(mean_monthly, 0.1), size=forecast_months)
                    simulated_totals.append(monthly_crashes.sum())
                
                simulated_totals = np.array(simulated_totals)
                
                # Calculate statistics
                mean_crashes = np.mean(simulated_totals)
                std_crashes = np.std(simulated_totals)
                percentile_5 = np.percentile(simulated_totals, 5)
                percentile_95 = np.percentile(simulated_totals, 95)
                percentile_25 = np.percentile(simulated_totals, 25)
                percentile_75 = np.percentile(simulated_totals, 75)
                
                # Display results
                st.success(f"Simulation complete! ({n_simulations:,} iterations)")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Expected Crashes", f"{mean_crashes:.1f}", help=f"Mean of {n_simulations:,} simulations")
                col2.metric("90% Confidence Interval", f"{percentile_5:.0f} - {percentile_95:.0f}")
                col3.metric("Standard Deviation", f"{std_crashes:.1f}")
                
                # Histogram of simulated results
                st.subheader("Probability Distribution of Predicted Crashes")
                
                fig = px.histogram(
                    x=simulated_totals,
                    nbins=50,
                    title=f'Distribution of Predicted Crashes ({forecast_months} months)',
                    labels={'x': 'Number of Crashes', 'y': 'Frequency'},
                    color_discrete_sequence=['#3498db']
                )
                
                # Add vertical lines for percentiles
                fig.add_vline(x=mean_crashes, line_dash="solid", line_color="red", 
                             annotation_text=f"Mean: {mean_crashes:.1f}")
                fig.add_vline(x=percentile_5, line_dash="dash", line_color="orange",
                             annotation_text=f"5th %ile: {percentile_5:.0f}")
                fig.add_vline(x=percentile_95, line_dash="dash", line_color="orange",
                             annotation_text=f"95th %ile: {percentile_95:.0f}")
                
                fig.update_layout(height=500, **get_plot_layout())
                st.plotly_chart(fig, width='stretch')
                
                show_analysis(f"Based on {n_simulations:,} Monte Carlo simulations, we predict {mean_crashes:.1f} crashes (±{std_crashes:.1f}) on {selected_street} over the next {forecast_months} months. There's a 90% probability the actual count will fall between {percentile_5:.0f} and {percentile_95:.0f} crashes.", "Prediction Summary")
                
                # Monthly breakdown simulation
                st.subheader("Simulated Monthly Crash Distribution")
                
                # Run a single representative simulation for visualization
                np.random.seed(42)
                sample_monthly = np.random.poisson(lam=max(mean_monthly, 0.1), size=forecast_months)
                
                months = pd.date_range(start=pd.Timestamp.now(), periods=forecast_months, freq='ME')
                monthly_sim_df = pd.DataFrame({
                    'Month': months.strftime('%Y-%m'),
                    'Predicted Crashes': sample_monthly,
                    'Lower Bound (25%)': np.random.poisson(lam=max(mean_monthly * 0.7, 0.1), size=forecast_months),
                    'Upper Bound (75%)': np.random.poisson(lam=max(mean_monthly * 1.3, 0.1), size=forecast_months)
                })
                
                fig = px.bar(
                    monthly_sim_df,
                    x='Month',
                    y='Predicted Crashes',
                    title='Sample Monthly Crash Prediction',
                    color_discrete_sequence=['#e74c3c']
                )
                fig.update_layout(height=400, **get_plot_layout())
                st.plotly_chart(fig, width='stretch')
                
                # Injury/Fatality risk estimation
                st.subheader("Casualty Risk Estimation")
                
                # Calculate historical injury/fatality rates from full dataset
                injury_rate = location_df_full['TOTAL_INJURED'].sum() / max(len(location_df_full), 1)
                fatality_rate = location_df_full['TOTAL_KILLED'].sum() / max(len(location_df_full), 1)
                
                # Estimate casualties based on predicted crashes
                predicted_injuries = mean_crashes * injury_rate
                predicted_fatalities = mean_crashes * fatality_rate
                
                col1, col2 = st.columns(2)
                col1.metric("Predicted Injuries", f"{predicted_injuries:.1f}", 
                           help=f"Based on historical injury rate of {injury_rate:.2f} per crash")
                col2.metric("Predicted Fatalities", f"{predicted_fatalities:.2f}",
                           help=f"Based on historical fatality rate of {fatality_rate:.4f} per crash")
                
                show_analysis(f"Based on historical rates, we estimate approximately {predicted_injuries:.1f} injuries and {predicted_fatalities:.2f} fatalities over the {forecast_months}-month forecast period. These estimates assume similar conditions to historical patterns.", "Casualty Forecast")
            else:
                st.warning("Insufficient historical data for this location. Please select a different street.")
    
    # Historical trend for context
    st.markdown("---")
    st.subheader("Historical Crash Trend for Selected Location")
    
    if len(location_df) > 0:
        yearly_location = location_df.groupby('YEAR').size().reset_index(name='Crashes')
        
        fig = px.line(yearly_location, x='YEAR', y='Crashes', markers=True,
                      title=f'Annual Crashes on {selected_street}')
        fig.update_layout(height=400, **get_plot_layout())
        st.plotly_chart(fig, width='stretch')
        
        # Hourly pattern
        col1, col2 = st.columns(2)
        
        with col1:
            hourly_location = location_df.groupby('HOUR').size().reset_index(name='Crashes')
            fig = px.bar(hourly_location, x='HOUR', y='Crashes',
                         title='Crashes by Hour of Day',
                         color='Crashes', color_continuous_scale=BLUE_SCALE)
            fig.update_layout(height=350, **get_plot_layout())
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            daily_location = location_df.groupby('DAY_NAME').size().reset_index(name='Crashes')
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            daily_location['DAY_NAME'] = pd.Categorical(daily_location['DAY_NAME'], categories=day_order, ordered=True)
            daily_location = daily_location.sort_values('DAY_NAME')
            
            fig = px.bar(daily_location, x='DAY_NAME', y='Crashes',
                         title='Crashes by Day of Week',
                         color='Crashes', color_continuous_scale=BLUE_SCALE)
            fig.update_layout(height=350, **get_plot_layout())
            st.plotly_chart(fig, width='stretch')
    else:
        st.info("No crash data available for the selected location and filters.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; line-height: 1.2;'>
        <p style='margin: 0.2rem 0;'>NYC Motor Vehicle Collisions Dashboard | Data Source: NYC Open Data</p>
        <p style='margin: 0.2rem 0;'>Data Preparation and Analysis by Li Fan, January 2026</p>
        <p style='margin: 0.2rem 0;'>Built with Streamlit & Plotly</p>
    </div>
    """, 
    unsafe_allow_html=True
)
