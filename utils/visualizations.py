import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd

def plt_scatter(df, x: str, y: str):
    df.plot(kind = 'scatter', x = x , y = y )
    plt.show()



def plt_hist(df, col):
    df[col].plot(kind = 'hist')
    plt.show()



def plt_box(df, col):
    df[col].plot(kind = 'box', figsize = (20, 6))
    plt.show()




def sns_countplot_of_cols(df, cols):
    sns.set_style('whitegrid')

    for feature in cols:
        if feature in df.columns:
            plt.figure(figsize=(20,10))
            sns.countplot(x=feature, data = df)
            plt.title(f'Count plot of {feature}')
            plt.xlabel(feature)
            plt.ylabel('Count')
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.show()


def plt_distributionplot(df,title: str, col):
    plt.figure(figsize = (10, 8))
    plt.style.use("fivethirtyeight")
    plt.title(title)
    sns.distplot(df[col])
    plt.show()


def px_scatter_with_trendline(df, x: str , y:str, size: str, title: str):
    figure = px.scatter(data_frame= df, x = x , y = y , size = size,
                        trendline= "ols", 
                        title= title)
    figure.show()



def plt_pie(df):
    df.plot(kind= "pie", figsize= (10, 8))
    plt.show()






def plt_pie2(df, title):
    """
    Plots a pie chart from a pandas Series (e.g., from value_counts()).
    """
    labels = df.index
    sizes = df.values

    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140)
    plt.axis("equal")  
    plt.title(title)
    plt.show()



def px_histogram(df, x_column,color_column, title ):
    fig_1 = px.histogram(df, x= x_column, color= color_column, title= title )
    fig_1.show()




def sns_plot_with_kde(data, x, title:str, y = None, ):
    plt.figure(figsize = (10, 8))
    sns.histplot(data=data, x=x, kde=True)
    plt.title(title)
    plt.show()




def plot_impressions_over_time(df: pd.DataFrame, date_col: str, target_col: str):
    plt.figure(figsize=(10, 5))
    df_sorted = df.sort_values(by=date_col)
    sns.lineplot(data=df_sorted, x=date_col, y=target_col, marker='o')
    plt.title("Impressions Over Time")
    plt.xlabel("Date")
    plt.ylabel("Impressions")
    plt.grid(True)
    plt.show()

def plot_avg_impressions_by_content(df: pd.DataFrame, content_col: str, target_col: str):
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x=content_col, y=target_col, estimator='mean', ci=None)
    plt.title("Average Impressions by Content Type")
    plt.xlabel("Content Type")
    plt.ylabel("Average Impressions")
    plt.show()


def plot_hashtags_vs_impressions(df: pd.DataFrame, hashtag_count_col: str, target_col: str):
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x=hashtag_count_col, y=target_col, alpha=0.7)
    plt.title("Hashtags vs Impressions")
    plt.xlabel("Number of Hashtags")
    plt.ylabel("Impressions")
    plt.grid(True)
    plt.show()


def plot_caption_length_vs_impressions(df: pd.DataFrame, caption_length_col: str, target_col: str):
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x=caption_length_col, y=target_col, alpha=0.7)
    plt.title("Caption Length vs Impressions")
    plt.xlabel("Caption Length (words)")
    plt.ylabel("Impressions")
    plt.grid(True)
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame, cols: list):
    plt.figure(figsize=(16, 14))
    corr = df[cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.show()

def plot_engagement_rate_distribution(df: pd.DataFrame, engagement_rate_col: str):
    plt.figure(figsize=(8, 5))
    sns.histplot(df[engagement_rate_col], bins=20, kde=True)
    plt.title("Engagement Rate Distribution")
    plt.xlabel("Engagement Rate")
    plt.ylabel("Frequency")
    plt.show()

def ols_plot(data,column_x, column_y, title):
    figure = px.scatter(data , x=column_x, y =column_y, size= column_y,
                    trendline= "ols", 
                    title= title)
    figure.show()