import sys, os
import numpy as np
import pandas as pd
import joblib
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../data')))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../models')))


@st.cache_resource
def load_model(path="models/ml_models/impression_model.pkl"):
    return joblib.load(path)

def predict():
    model = load_model()
    st.title("Instagram Impressions Predictor")
    st.markdown("Fill in the post engagement details below to forecast total **Impressions**.")
    caption = st.text_area("Caption", "Launching our new project...")
    hashtags = st.text_area("Hashtags", "#datascience #ml #python")
    likes = st.number_input("Likes", min_value=0, value=120)
    comments = st.number_input("Number of Comments", min_value=0, value=15)
    shares = st.number_input("Shares", min_value=0, value=8)
    saves = st.number_input("Saves", min_value=0, value=25)
    profile_visits = st.number_input("Profile Visits", min_value=0, value=60)

    day_of_week = st.selectbox("Day of Week (0=Mon … 6=Sun)", list(range(7)), index=0)
    hour_of_day = st.number_input("Hour of Day (0-23)", min_value=0, max_value=23, value=14)
    month = st.number_input("Month (1-12)", min_value=1, max_value=12, value=8)
    content_type = st.selectbox("Content Type", ["Photo", "Video", "Carousel"], index=0)
    is_weekend = st.radio("Is this a Weekend?", ["No", "Yes"], index=0)



    caption_length = len(caption.strip())
    hashtag_list = [tag for tag in hashtags.split() if tag.startswith("#")]
    hashtag_count = len(hashtag_list)
    hashtag_density = hashtag_count / caption_length if caption_length > 0 else 0
        
    row = {
    "CaptionLength": caption_length,
    "HashtagCount": hashtag_count,
    "HashtagDensity": hashtag_density,
    "Likes": likes,
    "Comments": comments,
    "Shares": shares,
    "Saves": saves,
    "Profile Visits": profile_visits,
    "DayOfWeek": day_of_week,
    "HourOfDay": hour_of_day,
    "Month": month,
    "IsWeekend": 1 if is_weekend == "Yes" else 0}
    
    
    instance = pd.DataFrame([row])

    if st.button("Predict Impressions"):
        pred = model.predict(instance)
        pred = np.expm1(pred)  


        st.markdown(
        f"""
        <div style="background-color:#e6f9e6; padding:20px; border-radius:12px; text-align:center;">
            <h1 style="color:green; font-size:48px; margin:0;">{int(pred[0]):,}</h1>
            <p style="color:#2e7d32; font-size:20px; margin:0;">Predicted Impressions</p>
        </div>
        """,
        unsafe_allow_html=True
    )
       