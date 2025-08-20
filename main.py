import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from datetime import datetime, timedelta
import warnings
from ui.overview import show_overview
from ui.time_series_analysis import show_time_analysis
from ui.content_performance import show_content_performance
from ui.engagement_analysis import show_engagement_analysis
from ui.advanced_analytics import show_advanced_analytics
from ui.inference import predict
warnings.filterwarnings('ignore')


st.set_page_config(
    page_title="Instagram Reach Analysis and Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stMetric {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the data"""
    try:
        # loads the data
        df = pd.read_csv('data/processed/clean_data.csv')
        
        # converts timestamp to datetime
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        
        # feature engineering 
        df['Date'] = df['Timestamp'].dt.date
        df['Hour'] = df['Timestamp'].dt.hour
        df['DayOfWeek'] = df['Timestamp'].dt.day_name()
        df['IsWeekend'] = df['Timestamp'].dt.dayofweek.isin([5, 6]).astype(int) 
        df['Month'] = df['Timestamp'].dt.month_name()
        df['MonthNum'] = df['Timestamp'].dt.month
        df['Year'] = df['Timestamp'].dt.year
        df['Quarter'] = df['Timestamp'].dt.quarter
        
        # calculates metrics
        df['TotalEngagement'] = df['Likes'] + df['Comments'] + df['Shares'] + df['Saves']
        df['EngagementPerImpression'] = df['TotalEngagement'] / df['Impressions']
        df['ConversionRate'] = df['Follows'] / df['Profile Visits']
        df['ConversionRate'] = df['ConversionRate'].fillna(0)
        
        # calculate reach efficiency
        df['ReachEfficiency'] = df['Profile Visits'] / df['Impressions']
        
        # content performance score 
        df['ContentScore'] = (
            df['Likes'] * 1 + 
            df['Comments'] * 3 + 
            df['Shares'] * 5 + 
            df['Saves'] * 4 + 
            df['Follows'] * 10
        ) / df['Impressions']
        
        # traffic source diversity (entropy-based)
        traffic_cols = ['From Home', 'From Hashtags', 'From Explore', 'From Other']
        df['TrafficDiversity'] = df[traffic_cols].apply(
            lambda row: -sum([p * np.log(p + 1e-10) for p in row / (row.sum() + 1e-10)]), axis=1
        )
        
        # Calculate traffic source ratios
        traffic_sum = df[traffic_cols].sum(axis=1)
        # Avoid division by zero by replacing 0 with 1 (ratio will be 0 anyway)
        safe_traffic_sum = traffic_sum.replace(0, 1)
        df['FromHomeRatio'] = df['From Home'] / safe_traffic_sum
        df['FromHashtagsRatio'] = df['From Hashtags'] / safe_traffic_sum
        df['FromExploreRatio'] = df['From Explore'] / safe_traffic_sum
        df['FromOtherRatio'] = df['From Other'] / safe_traffic_sum
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def main():
    # Title and header
    st.markdown('<h1 class="main-header">Instagram Reach Analysis and Prediction Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    if df is None:
        st.stop()
    
    # Sidebar filters
    st.sidebar.title(" Filters & Controls")
    
    # Date range filter
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(df['Date'].min(), df['Date'].max()),
        min_value=df['Date'].min(),
        max_value=df['Date'].max()
    )
    
    # Content type filter
    content_types = st.sidebar.multiselect(
        "Content Type",
        options=df['Content Type'].unique(),
        default=df['Content Type'].unique()
    )
    
    # Engagement threshold
    min_engagement = st.sidebar.slider(
        "Minimum Total Engagement",
        min_value=int(df['TotalEngagement'].min()),
        max_value=int(df['TotalEngagement'].max()),
        value=int(df['TotalEngagement'].min())
    )
    
    # Filter data
    if len(date_range) == 2:
        mask = (
            (pd.to_datetime(df['Date']).dt.date >= date_range[0]) & 
            (pd.to_datetime(df['Date']).dt.date <= date_range[1]) &
            (df['Content Type'].isin(content_types)) &
            (df['TotalEngagement'] >= min_engagement)
        )
        filtered_df = df[mask].copy()
    else:
        filtered_df = df.copy()
    
    if filtered_df.empty:
        st.warning("No data available for the selected filters.")
        return
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🔴 Overview", "🟢 Time Analysis", "🔵 Content Performance", 
        "🟠 Engagement Deep Dive", "🟡 Advanced Analytics", "🟣 Predictions"
    ])
    
    with tab1:
        show_overview(filtered_df)
    
    with tab2:
        show_time_analysis(filtered_df)
    
    with tab3:
        show_content_performance(filtered_df)
    
    with tab4:
        show_engagement_analysis(filtered_df)
    
    with tab5:
        show_advanced_analytics(filtered_df)
    with tab6:
        predict()
        


if __name__ == "__main__":
    main()
