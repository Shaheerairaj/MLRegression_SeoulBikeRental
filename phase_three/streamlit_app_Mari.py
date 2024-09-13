import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import joblib

#Center title
st.markdown("<h1 style='text-align: center;'>🌆 SEOUL BIKE RENTAL PREDICTOR 🚴‍♀️</h1>", unsafe_allow_html=True)

st.markdown("---")

#CSS customization
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f2f6;
    }

    .block-container {
        text-align: center;
    }

    .stTextInput, .stNumberInput, .stDateInput, .stRadio, .stSelectbox {
        margin: 0 auto;
        color: #333333;
        font-weight: bold;
    }

    label {
        display: flex;
        justify-content: center;
    }

    .stNumberInput input, .stDateInput input, .stTextInput input {
        background-color: #f7f9fc;
        border: 2px solid #4CAF50;
        border-radius: 8px;
    }

    .stRadio div[role="radiogroup"], .stSelectbox div[data-baseweb="select"] {
        background-color: #f7f9fc;
        border: 2px solid #4CAF50;
        border-radius: 8px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

#Define the preprocessing function
def preprocess_data(data):
    # Columns to drop if they exist
    columns_to_drop = ['Functioning Day', 'Dew point temperature(°C)']
    columns_to_drop = [col for col in columns_to_drop if col in data.columns]

    # Drop specified columns
    data = data.drop(columns=columns_to_drop, errors='ignore')

    # Derive date-based features
    data['Datetime'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')
    data['Day'] = data['Datetime'].dt.day
    data['Month'] = data['Datetime'].dt.month
    data['Day of Week'] = data['Datetime'].dt.dayofweek

    # Drop the original Date and Datetime columns
    data.drop(columns=['Datetime', 'Date'], inplace=True)

    return data 

#Load the trained model
@st.cache_resource
def load_model():
    model = joblib.load('./phase_three/Model_pipeline_Mari.pkl')
    return model

#Create column layouts
col1, col2 = st.columns(2)

with col1:
    st.markdown("<h4 style='text-align: center;'>Date and Time ⏰</h4>", unsafe_allow_html=True)
    user_inputs = {}
    # User input with specific date format "DD/MM/YYYY"
    user_inputs['Date'] = st.date_input("Enter Date (DD/MM/YYYY):", value=datetime.now(), format="DD/MM/YYYY")
    user_inputs['Hour'] = st.number_input("Select Hour (24-hour format):", min_value=0, max_value=23, value=12, step=1)

    st.markdown("<h4 style='text-align: center;'>Weather Information 🌤</h4>", unsafe_allow_html=True)
    user_inputs['Temperature(°C)'] = st.number_input("Temperature (°C):", min_value=-20.0, max_value=50.0, value=15.0, step=0.1, format="%.1f")
    user_inputs['Humidity(%)'] = st.number_input("Humidity (%):", min_value=10, max_value=100, value=50, step=1)

with col2:
    st.markdown("<h4 style='text-align: center;'>Environmental Conditions 🌍</h4>", unsafe_allow_html=True)
    user_inputs['Wind speed (m/s)'] = st.number_input("Wind speed (m/s):", min_value=0.0, max_value=20.0, value=5.0, step=0.1, format="%.1f")
    user_inputs['Visibility (10m)'] = st.number_input("Visibility (10m):", min_value=0, max_value=2000, value=1000, step=1)
    user_inputs['Solar Radiation (MJ/m2)'] = st.number_input("Solar Radiation (MJ/m2):", min_value=0.0, max_value=5.0, value=2.5, step=0.1, format="%.1f")
    user_inputs['Rainfall(mm)'] = st.number_input("Rainfall (mm):", min_value=0.0, max_value=200.0, value=0.0, step=0.1, format="%.1f")
    user_inputs['Snowfall (cm)'] = st.number_input("Snowfall (cm):", min_value=0.0, max_value=50.0, value=0.0, step=0.1, format="%.1f")

st.markdown("---")

st.markdown("<h4 style='text-align: center;'>Additional Information 🗓</h4>", unsafe_allow_html=True)
col3, col4 = st.columns(2)

with col3:
    seasons = ["Spring", "Summer", "Autumn", "Winter"]
    user_inputs['Seasons'] = st.selectbox("Select Season:", seasons)

with col4:
    holiday_options = ["No Holiday", "Holiday"]
    user_inputs['Holiday'] = st.radio("Is it a holiday?", holiday_options)

#Convert the user inputs into a DataFrame
input_df = pd.DataFrame([user_inputs])

#Preprocess the data
processed_data = preprocess_data(input_df)

#Load the model
model = load_model()

st.markdown("---")

#Button to trigger prediction
if st.button("Predict"):
    # Predict using the model
    try:
        prediction = np.expm1(model.predict(processed_data))
        st.success(f"**Predicted 🚴‍♀️ Rentals**: {int(max(prediction[0], 0))}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
