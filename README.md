# instagram-reach-analysis
---
##  Instagram Reach Analysis and Prediction

Predict Instagram **Impressions** from early engagement signals and post metadata.  
Built with scikit-learn pipelines (feature engineering + model), TF-IDF for text when needed, and a Streamlit app for simple and fast inference.

## Project Overview
---
**Goal**: Analyze and visualize the factors that impact user engagement and content reach on Instagram.


- Cleans and engineers features from raw Instagram post data.
- Trains  models (includes regularized linear models, tree boosting, and an ensemble).
- Uses **log1p(Impressions)** as the target for stability.
- Evaluates with repeated stratified CV on binned targets to reduce variance on small data.
- Serves predictions and Insight with visualizations via **Streamlit**.
- Saves & loads the **entire pipeline** (preprocessing + model) for reproducible inference.


##  Key Features
---

- Data Cleaning & Preprocessing
- Exploratory Data Analysis (EDA)
- Visualizations using:
  - Seaborn
  - Matplotlib
  - Plotly
  - WordClouds
- Correlation analysis
- Feature engineering and selection

## Libraries Used
---

- See `requirements.txt` file for the libraries used 



## How to Run
---

- Clone the repository using `git clone <repo link>`
- Then type `streamlit run main.py` in your terminal


##  Example Outputs
---

- Heatmaps and bar charts showing engagement metrics
- Time-Series Charts
- Visual correlations between features and reach

##  Author
---

- Ayomide Adegoke
- dynamic.ayo100@gmail.com






