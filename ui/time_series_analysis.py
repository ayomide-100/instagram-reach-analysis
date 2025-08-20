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





def show_time_analysis(df):
    """time-based analysis ui"""
    st.header(" Temporal Performance Analysis")
    
    
    st.subheader(" Performance Trends Over Time")
    
    
    daily_metrics = df.groupby('Date').agg({
        'Impressions': 'sum',
        'TotalEngagement': 'sum',
        'Profile Visits': 'sum',
        'Follows': 'sum',
        'EngagementRate': 'mean'
    }).reset_index()
    
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Daily Impressions', 'Daily Engagement', 'Profile Visits', 'Follows'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    
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
    
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(" Best Posting Hours")
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
        st.subheader(" Day of Week Performance")
        
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
    
    
    st.subheader(" Monthly Performance Trends")
    monthly_data = df.groupby(['Year', 'MonthNum']).agg({
        'Impressions': 'sum',
        'TotalEngagement': 'sum',
        'EngagementRate': 'mean',
        'Follows': 'sum'
    }).reset_index()
    
    
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
