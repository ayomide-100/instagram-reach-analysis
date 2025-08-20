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
warnings.filterwarnings('ignore')



def show_advanced_analytics(df):
    """Advanced analytics and insights"""
    st.header(" Advanced Analytics & Insights")
    
    # Performance prediction model (simple)
    st.subheader(" Performance Insights")
    
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
    st.subheader(" Optimization Recommendations")
    
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
    st.subheader(" Trend Analysis")
    
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
    st.subheader(" Statistical Summary")
    
    summary_stats = df[['Impressions', 'TotalEngagement', 'EngagementRate', 
                       'Profile Visits', 'Follows', 'ContentScore']].describe().round(2)
    st.dataframe(summary_stats, use_container_width=True)
