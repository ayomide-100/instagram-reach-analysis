import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Instagram Reach Analysis and Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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
        # Load the CSV file
        df = pd.read_csv('data/processed/clean_data.csv')
        
        # Convert timestamp to datetime
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        
        # Feature Engineering
        df['Date'] = df['Timestamp'].dt.date
        df['Hour'] = df['Timestamp'].dt.hour
        df['DayOfWeek'] = df['Timestamp'].dt.day_name()
        df['IsWeekend'] = df['Timestamp'].dt.dayofweek.isin([5, 6]).astype(int) # 5=Sat, 6=Sun
        df['Month'] = df['Timestamp'].dt.month_name()
        df['MonthNum'] = df['Timestamp'].dt.month
        df['Year'] = df['Timestamp'].dt.year
        df['Quarter'] = df['Timestamp'].dt.quarter
        
        # Calculate additional metrics
        df['TotalEngagement'] = df['Likes'] + df['Comments'] + df['Shares'] + df['Saves']
        df['EngagementPerImpression'] = df['TotalEngagement'] / df['Impressions']
        df['ConversionRate'] = df['Follows'] / df['Profile Visits']
        df['ConversionRate'] = df['ConversionRate'].fillna(0)
        
        # Calculate reach efficiency
        df['ReachEfficiency'] = df['Profile Visits'] / df['Impressions']
        
        # Content performance score (normalized)
        df['ContentScore'] = (
            df['Likes'] * 1 + 
            df['Comments'] * 3 + 
            df['Shares'] * 5 + 
            df['Saves'] * 4 + 
            df['Follows'] * 10
        ) / df['Impressions']
        
        # Traffic source diversity (entropy-based)
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
    st.markdown('<h1 class="main-header">📊 Social Media Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    if df is None:
        st.stop()
    
    # Sidebar filters
    st.sidebar.title("🔧 Filters & Controls")
    
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
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Overview", "⏰ Time Analysis", "🎯 Content Performance", 
        "🚀 Engagement Deep Dive", "🔍 Advanced Analytics"
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

def show_overview(df):
    """Overview dashboard section"""
    st.header("📊 Key Performance Metrics")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Impressions",
            f"{df['Impressions'].sum():,}",
            delta=f"{df['Impressions'].mean():.0f} avg"
        )
    
    with col2:
        st.metric(
            "Total Engagement",
            f"{df['TotalEngagement'].sum():,}",
            delta=f"{df['EngagementRate'].mean():.2%} rate"
        )
    
    with col3:
        st.metric(
            "Profile Visits",
            f"{df['Profile Visits'].sum():,}",
            delta=f"{df['ReachEfficiency'].mean():.2%} efficiency"
        )
    
    with col4:
        st.metric(
            "Total Follows",
            f"{df['Follows'].sum():,}",
            delta=f"{df['ConversionRate'].mean():.2%} conversion"
        )
    
    # Main overview charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Impressions vs Engagement Over Time")
        fig = px.scatter(
            df, x='Impressions', y='TotalEngagement', 
            size='Profile Visits', color='Content Type',
            hover_data=['Timestamp', 'EngagementRate'],
            title="Content Performance Bubble Chart"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Content Type Performance")
        content_metrics = df.groupby('Content Type').agg({
            'Impressions': 'mean',
            'TotalEngagement': 'mean',
            'EngagementRate': 'mean',
            'Follows': 'mean'
        }).round(2)
        
        fig = px.bar(
            content_metrics.reset_index(),
            x='Content Type', y='EngagementRate',
            title="Average Engagement Rate by Content Type",
            color='EngagementRate',
            color_continuous_scale='viridis'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Traffic source analysis
    st.subheader("🌊 Traffic Source Distribution")
    traffic_cols = ['From Home', 'From Hashtags', 'From Explore', 'From Other']
    traffic_totals = df[traffic_cols].sum()
    
    fig = px.pie(
        values=traffic_totals.values,
        names=traffic_totals.index,
        title="Overall Traffic Source Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

def show_time_analysis(df):
    """Time-based analysis section"""
    st.header("⏰ Temporal Performance Analysis")
    
    # Time series plot
    st.subheader("📈 Performance Trends Over Time")
    
    # Aggregate daily metrics
    daily_metrics = df.groupby('Date').agg({
        'Impressions': 'sum',
        'TotalEngagement': 'sum',
        'Profile Visits': 'sum',
        'Follows': 'sum',
        'EngagementRate': 'mean'
    }).reset_index()
    
    # Create subplot
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Daily Impressions', 'Daily Engagement', 'Profile Visits', 'Follows'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Add traces
    fig.add_trace(
        go.Scatter(x=daily_metrics['Date'], y=daily_metrics['Impressions'], 
                  name='Impressions', line=dict(color='blue')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=daily_metrics['Date'], y=daily_metrics['TotalEngagement'], 
                  name='Engagement', line=dict(color='green')),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(x=daily_metrics['Date'], y=daily_metrics['Profile Visits'], 
                  name='Profile Visits', line=dict(color='orange')),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=daily_metrics['Date'], y=daily_metrics['Follows'], 
                  name='Follows', line=dict(color='red')),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Hourly and daily patterns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🕐 Best Posting Hours")
        hourly_performance = df.groupby('Hour').agg({
            'Impressions': 'mean',
            'EngagementRate': 'mean',
            'TotalEngagement': 'mean'
        }).reset_index()
        
        fig = px.line(
            hourly_performance, x='Hour', y='EngagementRate',
            title="Average Engagement Rate by Hour",
            markers=True
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📅 Day of Week Performance")
        # Reorder days properly
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        df['DayOfWeek'] = pd.Categorical(df['DayOfWeek'], categories=day_order, ordered=True)
        
        daily_performance = df.groupby('DayOfWeek').agg({
            'Impressions': 'mean',
            'EngagementRate': 'mean',
            'TotalEngagement': 'mean'
        }).reset_index()
        
        fig = px.bar(
            daily_performance, x='DayOfWeek', y='TotalEngagement',
            title="Average Engagement by Day of Week",
            color='TotalEngagement',
            color_continuous_scale='plasma'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Monthly trends
    st.subheader("📆 Monthly Performance Trends")
    monthly_data = df.groupby(['Year', 'MonthNum']).agg({
        'Impressions': 'sum',
        'TotalEngagement': 'sum',
        'EngagementRate': 'mean',
        'Follows': 'sum'
    }).reset_index()
    
    # Create a date column for proper sorting
    # FIX: Construct datetime from a dictionary with lowercase keys
    monthly_data['Date'] = pd.to_datetime(dict(year=monthly_data.Year, month=monthly_data.MonthNum, day=1))
    monthly_data = monthly_data.sort_values('Date')
    
    # Add month names for better labels
    monthly_data['MonthName'] = monthly_data['Date'].dt.month_name()
    monthly_data['YearMonth'] = monthly_data['Year'].astype(str) + '-' + monthly_data['MonthName']
    
    fig = px.line(
        monthly_data, x='Date', y=['Impressions', 'TotalEngagement'],
        title="Monthly Impressions vs Engagement Trends",
        labels={'Date': 'Month-Year'}
    )
    fig.update_layout(xaxis_tickformat='%Y-%m')
    st.plotly_chart(fig, use_container_width=True)

def show_content_performance(df):
    """Content performance analysis section"""
    st.header("🎯 Content Performance Analysis")
    
    # Top performing posts
    st.subheader("🏆 Top Performing Posts")
    
    metric_choice = st.selectbox(
        "Sort by:",
        ['TotalEngagement', 'EngagementRate', 'Impressions', 'ContentScore', 'Follows']
    )
    
    top_posts = df.nlargest(10, metric_choice)[
        ['Caption', 'Content Type', 'Impressions', 'TotalEngagement', 
         'EngagementRate', 'Follows', 'Timestamp']
    ].round(4)
    
    st.dataframe(top_posts, use_container_width=True)
    
    # Content length analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Caption Length vs Performance")
        fig = px.scatter(
            df, x='CaptionLength', y='EngagementRate',
            color='Content Type', size='Impressions',
            title="Caption Length Impact on Engagement",
            hover_data=['TotalEngagement']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🏷️ Hashtag Performance")
        fig = px.scatter(
            df, x='HashtagCount', y='EngagementRate',
            color='Content Type', size='Impressions',
            title="Hashtag Count vs Engagement Rate",
            hover_data=['TotalEngagement']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Content score distribution
    st.subheader("⭐ Content Score Distribution")
    fig = px.histogram(
        df, x='ContentScore', color='Content Type',
        title="Distribution of Content Performance Scores",
        marginal="box"
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

def show_engagement_analysis(df):
    """Detailed engagement analysis"""
    st.header("🚀 Engagement Deep Dive")
    
    # Engagement breakdown
    st.subheader("💫 Engagement Type Analysis")
    
    engagement_cols = ['Likes', 'Comments', 'Shares', 'Saves']
    engagement_totals = df[engagement_cols].sum()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.pie(
            values=engagement_totals.values,
            names=engagement_totals.index,
            title="Engagement Type Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Engagement ratios
        engagement_ratios = df[engagement_cols].div(df['TotalEngagement'] + 1e-10, axis=0).mean()
        fig = px.bar(
            x=engagement_ratios.index, y=engagement_ratios.values,
            title="Average Engagement Type Ratios",
            color=engagement_ratios.values,
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    st.subheader("🔗 Correlation Analysis")
    numeric_cols = ['Impressions', 'Likes', 'Comments', 'Shares', 'Saves', 
                   'Profile Visits', 'Follows', 'CaptionLength', 'HashtagCount']
    correlation_matrix = df[numeric_cols].corr()
    
    fig = px.imshow(
        correlation_matrix,
        title="Feature Correlation Matrix",
        color_continuous_scale='RdBu_r',
        aspect='auto'
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Engagement rate by content characteristics
    st.subheader("📊 Engagement Patterns")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Weekend vs weekday performance
        weekend_comparison = df.groupby('IsWeekend').agg({
            'EngagementRate': 'mean',
            'TotalEngagement': 'mean',
            'Impressions': 'mean'
        }).reset_index()
        weekend_comparison['IsWeekend'] = weekend_comparison['IsWeekend'].map({0: 'Weekday', 1: 'Weekend'})
        
        fig = px.bar(
            weekend_comparison, x='IsWeekend', y='EngagementRate',
            title="Weekend vs Weekday Performance",
            color='EngagementRate',
            color_continuous_scale='blues'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Traffic source efficiency
        traffic_efficiency = df.groupby('Content Type').agg({
            'FromHomeRatio': 'mean',
            'FromHashtagsRatio': 'mean',
            'FromExploreRatio': 'mean',
            'FromOtherRatio': 'mean'
        }).reset_index()
        
        fig = px.bar(
            traffic_efficiency.melt(id_vars='Content Type'),
            x='Content Type', y='value', color='variable',
            title="Traffic Source Distribution by Content Type",
            barmode='stack'
        )
        st.plotly_chart(fig, use_container_width=True)

def show_advanced_analytics(df):
    """Advanced analytics and insights"""
    st.header("🔍 Advanced Analytics & Insights")
    
    # Performance prediction model (simple)
    st.subheader("🎯 Performance Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Content performance segmentation
        df['PerformanceSegment'] = pd.qcut(
            df['ContentScore'], 
            q=3, 
            labels=['Low', 'Medium', 'High']
        )
        
        segment_analysis = df.groupby('PerformanceSegment').agg({
            'CaptionLength': 'mean',
            'HashtagCount': 'mean',
            'EngagementRate': 'mean',
            'Impressions': 'mean'
        }).round(2)
        
        st.write("**Performance Segment Characteristics:**")
        st.dataframe(segment_analysis)
    
    with col2:
        # Traffic diversity analysis
        fig = px.box(
            df, x='Content Type', y='TrafficDiversity',
            title="Traffic Source Diversity by Content Type"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Optimal posting recommendations
    st.subheader("💡 Optimization Recommendations")
    
    # Find best performing combinations
    best_hour = df.loc[df['EngagementRate'].idxmax(), 'Hour']
    best_day = df.groupby('DayOfWeek')['EngagementRate'].mean().idxmax()
    best_content_type = df.groupby('Content Type')['EngagementRate'].mean().idxmax()
    
    optimal_caption_length = df.loc[df['EngagementRate'].idxmax(), 'CaptionLength']
    optimal_hashtag_count = df.loc[df['EngagementRate'].idxmax(), 'HashtagCount']
    
    recommendations = {
        "Best Posting Hour": f"{best_hour}:00",
        "Best Day": best_day,
        "Top Content Type": best_content_type,
        "Optimal Caption Length": f"{optimal_caption_length} words",
        "Optimal Hashtag Count": f"{optimal_hashtag_count} hashtags"
    }
    
    col1, col2, col3, col4, col5 = st.columns(5)
    cols = [col1, col2, col3, col4, col5]

    for i, (key, value) in enumerate(recommendations.items()):
        with cols[i]:
            st.metric(key, value)
    
    # Trend analysis
    st.subheader("📈 Trend Analysis")
    
    # Calculate rolling averages
    df_sorted = df.sort_values('Timestamp')
    df_sorted['EngagementRate_MA7'] = df_sorted['EngagementRate'].rolling(window=7, min_periods=1).mean()
    df_sorted['Impressions_MA7'] = df_sorted['Impressions'].rolling(window=7, min_periods=1).mean()
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(x=df_sorted['Timestamp'], y=df_sorted['EngagementRate_MA7'],
                  name='Engagement Rate (7-day MA)', line=dict(color='blue')),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(x=df_sorted['Timestamp'], y=df_sorted['Impressions_MA7'],
                  name='Impressions (7-day MA)', line=dict(color='red')),
        secondary_y=True,
    )
    
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Engagement Rate", secondary_y=False)
    fig.update_yaxes(title_text="Impressions", secondary_y=True)
    fig.update_layout(title_text="Performance Trends (7-Day Moving Average)")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary statistics
    st.subheader("📋 Statistical Summary")
    
    summary_stats = df[['Impressions', 'TotalEngagement', 'EngagementRate', 
                       'Profile Visits', 'Follows', 'ContentScore']].describe().round(2)
    st.dataframe(summary_stats, use_container_width=True)

if __name__ == "__main__":
    main()
