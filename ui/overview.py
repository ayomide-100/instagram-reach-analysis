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



def show_overview(df):
    """overview dashboard ui"""
    st.header(" Key Performance Metrics")
    
    
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
    
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(" Impressions vs Engagement Over Time")
        fig = px.scatter(
            df, x='Impressions', y='TotalEngagement', 
            size='Profile Visits', color='Content Type',
            hover_data=['Timestamp', 'EngagementRate'],
            title="Content Performance Bubble Chart"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader(" Content Type Performance")
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
    st.subheader(" Traffic Source Distribution")
    traffic_cols = ['From Home', 'From Hashtags', 'From Explore', 'From Other']
    traffic_totals = df[traffic_cols].sum()
    
    fig = px.pie(
        values=traffic_totals.values,
        names=traffic_totals.index,
        title="Overall Traffic Source Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

