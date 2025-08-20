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

def show_content_performance(df):
    """Content performance analysis section"""
    st.header(" Content Performance Analysis")
    
    # Top performing posts
    st.subheader(" Top Performing Posts")
    
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
        st.subheader(" Caption Length vs Performance")
        fig = px.scatter(
            df, x='CaptionLength', y='EngagementRate',
            color='Content Type', size='Impressions',
            title="Caption Length Impact on Engagement",
            hover_data=['TotalEngagement']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader(" Hashtag Performance")
        fig = px.scatter(
            df, x='HashtagCount', y='EngagementRate',
            color='Content Type', size='Impressions',
            title="Hashtag Count vs Engagement Rate",
            hover_data=['TotalEngagement']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Content score distribution
    st.subheader(" Content Score Distribution")
    fig = px.histogram(
        df, x='ContentScore', color='Content Type',
        title="Distribution of Content Performance Scores",
        marginal="box"
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
