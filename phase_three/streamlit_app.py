import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
import os

try:
    import lightgbm as lgb
except ImportError:
    st.error("The lightgbm package is not installed. Please install it using 'pip install lightgbm' or make sure it's included in your requirements.txt file.")
    st.stop()

st.markdown("""
    <style>
    .stApp {
        background-color: #2F2F2F;
    }
    .main .block-container {
        color: white;
    }
    .stButton>button {
        color: black;
    }
    .stSelectbox>div>div>div {
        color: black;
    }
    .stSlider label {
        color: white !important;
    }
    h1, h2, h3, h4, h5, h6, .stMarkdown {
        color: white;
    }
            
    .sidebar .sidebar-content {
        color: black !important;
    }
    .sidebar .sidebar-content h1,
    .sidebar .sidebar-content h2,
    .sidebar .sidebar-content h3,
    .sidebar .sidebar-content p,
    .sidebar .sidebar-content div {
        color: black !important;
    }
            
    div[data-baseweb="select"] > div {
        color: white !important;
    }
            
    .prediction-box {
        background-color: white;
        color: black;
        padding: 10px;
        border-radius: 5px;
        margin-top: 20px;
    }
    .prediction-box .prediction-label {
        font-weight: bold;
    }
    .about-box {
        background-color: white;
        color: black;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .stApp header {
        background-color: #2F2F2F;
    }
    .info-box {
        background-color: white;
        color: black;
        padding: 15px;
        border-radius: 10px;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    .info-box a {
        color: #0000EE;
    }
    .white-label {
        color: white !important;
        margin-bottom: 0px !important;
        font-size: 14px;
        font-weight: bold;
    }
    .extra-space {
        margin-bottom: 20px;
    }
    .categorical-container {
        margin-bottom: 10px;
    }
    [data-testid="stSidebar"] [data-testid="stImage"] {
        border-radius: 10px;
        overflow: hidden;
    }
    </style>
    """, unsafe_allow_html=True)

try:
    model_path = os.path.join(os.path.dirname(__file__), "exported_model_timo.pkl")
    model, ref_cols, target = joblib.load(model_path)
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'exported_model_timo.pkl' is in the same directory as this app.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the model: {str(e)}")
    st.stop()

# Load and display the new image in the sidebar with rounded edges
try:
    image_path = os.path.join(os.path.dirname(__file__), "bike_rental_seoul.jpg")
    image = Image.open(image_path)
    st.sidebar.image(image, use_column_width=True)
except FileNotFoundError:
    st.sidebar.warning("Image file 'bike_rental_seoul.jpg' not found. Continuing without image.")
except Exception as e:
    st.sidebar.warning(f"An error occurred while loading the image: {str(e)}")

# Sidebar content
st.sidebar.markdown("""
    <div class="about-box">
        <h3 style="color: black;">About this application</h3>
        <p style="color: black;">This app uses a LightGBM model to predict the number of rented bikes in Seoul based on the features listed below</p>
    </div>
    """, unsafe_allow_html=True)

st.sidebar.markdown('<h3 style="color: black;">Features</h3>', unsafe_allow_html=True)

st.sidebar.markdown("""
    <p style="color: black;"><strong>Numeric features:</strong></p>
    <ul style="color: black;">
        <li>Hour</li>
        <li>Temperature(°C)</li>
        <li>Humidity(%)</li>
        <li>Wind speed (m/s)</li>
        <li>Visibility (10m)</li>
        <li>Solar Radiation (MJ/m2)</li>
        <li>Rainfall(mm)</li>
        <li>Snowfall (cm)</li>
    </ul>
    """, unsafe_allow_html=True)

st.sidebar.markdown("""
    <p style="color: black;"><strong>Categorical features:</strong></p>
    <ul style="color: black;">
        <li>Seasons</li>
        <li>Holiday</li>
        <li>DayType</li>
    </ul>
    """, unsafe_allow_html=True)

numeric_features = model.named_steps['preprocessor'].transformers_[0][2]
categorical_features = model.named_steps['preprocessor'].transformers_[1][2]

# Main section
st.title("Bike Rental Predictor")

# Add the dataset information box
st.markdown("""
    <div class="info-box">
        This predictor is based on the following dataset: <a href="https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand" target="_blank">Seoul Bike Sharing Demand</a>. The majority of datapoints were collected in the year 2018 with an overall of 8760 rows and 13 Features. Disclaimer: Feature engineering was performed.
    </div>
    """, unsafe_allow_html=True)

# Add the usage instructions box
st.markdown("""
    <div class="info-box">
        <strong>How to use this application:</strong>
        <ol>
            <li>Use the sliders to select your numerical values</li>
            <li>Select the categorical features from the drop downs</li>
            <li>Click on Predict Rented Bike Count</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

# Create input fields for each feature
input_data = {}

# Create sliders for numeric features with appropriate ranges
st.subheader("Numeric Features")
slider_ranges = {
    'Hour': (0, 23, 1),
    'Temperature(°C)': (-20.0, 40.0, 0.1),
    'Humidity(%)': (0, 100, 1),
    'Wind speed (m/s)': (0.0, 20.0, 0.1),
    'Visibility (10m)': (0, 2000, 10),
    'Solar Radiation (MJ/m2)': (0.0, 5.0, 0.01),
    'Rainfall(mm)': (0.0, 100.0, 0.1),
    'Snowfall (cm)': (0.0, 30.0, 0.1)
}

for col in numeric_features:
    min_val, max_val, step = slider_ranges.get(col, (0.0, 100.0, 1.0))
    if isinstance(min_val, int) and isinstance(max_val, int) and isinstance(step, int):
        value = int(min_val)
    else:
        min_val, max_val, step = float(min_val), float(max_val), float(step)
        value = float(min_val)
    input_data[col] = st.slider(f"Select {col}", min_value=min_val, max_value=max_val, value=value, step=step)

# Create input fields for categorical features
st.subheader("Categorical Features")
for col in categorical_features:
    if col == 'Holiday':
        options = ['No Holiday', 'Holiday']
    elif col == 'DayType':
        options = ['Weekend', 'Weekday']
    elif col == 'Seasons':
        options = ['Spring', 'Summer', 'Autumn', 'Winter']
    else:
        cat_index = list(categorical_features).index(col)
        options = model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].categories_[cat_index]
    
    st.markdown(f'<div class="categorical-container"><p class="white-label">Select {col}</p>', unsafe_allow_html=True)
    input_data[col] = st.selectbox("", options=options, key=f"categorical_{col}", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

# Add extra space after the last categorical feature
st.markdown('<div class="extra-space"></div>', unsafe_allow_html=True)

# Create a prediction button
if st.button("Predict Rented Bike Count"):
    input_df = pd.DataFrame([input_data])
    
    for col in categorical_features:
        input_df[col] = input_df[col].astype('category')
    
    prediction = model.predict(input_df)[0]
    prediction = max(0, prediction)
    
    st.markdown(f"""
    <div class="prediction-box">
        <span class="prediction-label">Predicted {target}:</span> {prediction:.2f}
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
    **Note on Predictions:**
    The model may occasionally produce negative predictions for certain input combinations. 
    These are automatically adjusted to 0, as negative bike rentals are not possible. 
    If you frequently see 0 predictions, it might indicate unusual input combinations or limitations in the model's training data.
    """)