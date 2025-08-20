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


def show_engagement_analysis(df):
    """Detailed engagement analysis"""
    st.header(" Engagement Deep Dive")
    
    # Engagement breakdown
    st.subheader(" Engagement Type Analysis")
    
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
    st.subheader(" Correlation Analysis")
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
    st.subheader(" Engagement Patterns")
    
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