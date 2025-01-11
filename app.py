import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
MODEL_PATH = "./models/xgb_model.pkl"
SCALER_PATH = "./scalers/scaler.pkl"
DATA_PATH = "./data/cleaned_station_data.csv"

with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)

with open(SCALER_PATH, "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

facility_type_mapping = {
    1: "Public Charging Station",
    2: "Private Charging Station",
    3: "Corporate Charging Station",
    4: "Residential Charging Station"
}


def preprocess_features(inputs):
    """
    Preprocesses input data to match the expected feature columns.
    """
    start_time = inputs["startTime"]
    end_time = inputs["endTime"]
    charge_time = inputs["chargeTimeHrs"]
    facility_type = inputs["facilityType"]
    created_hour = inputs["created_hour"]
    created_day = inputs["created_day"]
    created_month = inputs["created_month"]
    session_duration = inputs["session_duration"]
    kwh_per_hour = inputs["kwh_per_hour"]
    weekday = inputs["weekday"]
    start_period = inputs["startPeriod"]
    end_period = inputs["endPeriod"]

    # One-hot encode weekdays
    weekday_cols = ["Mon", "Sat", "Sun", "Thu", "Tue", "Wed"]
    weekday_encoding = [1 if day == weekday else 0 for day in weekday_cols]

    # One-hot encode periods
    period_cols = ["Evening", "Morning", "Night"]
    start_period_encoding = [1 if period == start_period else 0 for period in period_cols]
    end_period_encoding = [1 if period == end_period else 0 for period in period_cols]

    # Combine all features into a single array
    features = np.array([
        start_time,
        end_time,
        charge_time,
        facility_type,
        created_hour,
        created_day,
        created_month,
        session_duration,
        kwh_per_hour,
        *weekday_encoding,
        *start_period_encoding,
        *end_period_encoding,
    ])

    # Only scale the specified features: session_duration, kwh_per_hour
    features_to_scale = np.array([session_duration, kwh_per_hour])
    features_scaled = scaler.transform([features_to_scale])[0]

    # Replace the original features with the scaled ones
    features[7] = features_scaled[0]  # session_duration
    features[8] = features_scaled[1]  # kwh_per_hour

    return features.reshape(1, -1)

def create_visualizations(df):
    """
    Generate time series charts and visualizations for the dataset using original data.
    """
    # Energy consumption by hour
    st.subheader("Energy Consumption by Hour")
    hourly_consumption = df.groupby('created_hour')['kwh_per_hour'].mean().reset_index()
    fig = go.Figure(data=go.Scatter(x=hourly_consumption['created_hour'], 
                                   y=hourly_consumption['kwh_per_hour'], 
                                   mode='lines+markers', name='Energy Consumption'))
    fig.update_layout(title='Energy Consumption by Hour', 
                      xaxis_title='Hour of Day', 
                      yaxis_title='Average kWh per Hour')
    st.plotly_chart(fig)

    # Monthly trends
    st.subheader("Monthly Trends")
    monthly_trends = df.groupby('created_month')['kwh_per_hour'].mean().reset_index()
    fig = go.Figure(data=go.Scatter(x=monthly_trends['created_month'], 
                                   y=monthly_trends['kwh_per_hour'], 
                                   mode='lines+markers', name='Monthly Trends'))
    fig.update_layout(title='Monthly Trends in Energy Consumption',
                      xaxis_title='Month', 
                      yaxis_title='Average kWh per Hour')
    st.plotly_chart(fig)

    # Get counts of each facility type
    facility_usage = df['facilityType'].value_counts().reset_index()
    facility_usage.columns = ['facilityType', 'Count']

    # Map temporary labels for plotting
    facility_usage['Facility Type'] = facility_usage['facilityType'].map(facility_type_mapping)

    # Create the bar chart
    fig = go.Figure(
        data=go.Bar(
            x=facility_usage['Facility Type'], 
            y=facility_usage['Count'], 
            marker_color='red'
        )
    )

    fig.update_layout(
        title='Facility Type Usage Distribution',
        xaxis_title='Facility Type',
        yaxis_title='Count'
    )

    st.plotly_chart(fig)

def main():
    # Set up the page layout and inputs in the sidebar
    st.set_page_config(
        page_title="EV Charging Station Predictor",
        page_icon="🔋",
        layout="wide"
    )

    # Landing Page Title
    st.title("🔋 EV Charging Station Energy Consumption Predictor")
    st.markdown("""
    **Welcome to the EV Charging Station Predictor!**
    A powerful tool to optimize energy usage and enhance operational efficiency for EV charging stations.
    """)

    # Problem Statement Section
    st.header("🌍 Why This App is Needed")
    st.markdown("""
    The rapid adoption of electric vehicles (EVs) has brought challenges in managing energy consumption at charging stations. 
    Without accurate predictions, station operators face:
    - **Energy Overconsumption**: Higher costs and grid strain due to inaccurate energy estimates.
    - **Operational Inefficiency**: Difficulty in planning for peak usage hours.
    - **Customer Dissatisfaction**: Long wait times and energy shortages during peak periods.

    **This app provides predictive insights to address these issues**, empowering charging station operators to:
    - Forecast energy requirements for individual sessions.
    - Optimize resource allocation.
    - Improve the overall charging experience for EV users.
    """)

    # Key Features Section
    st.header("✨ Key Features")
    st.markdown("""
    - **Energy Consumption Prediction**: Get accurate kWh predictions for each charging session.
    - **Data Visualization**: Understand energy trends through interactive charts.
    - **Customizable Inputs**: Tailor predictions to specific scenarios with detailed input options.
    - **User-Friendly Design**: Easy-to-use interface with clear results and actionable insights.
    """)

    # How It Works Section
    st.header("🔍 How It Works")
    st.markdown("""
    1. Input session details such as start time, charge duration, and facility type.
    2. Click on **Predict Energy Consumption**.
    3. Get an instant prediction along with visual insights to optimize energy usage.
    """)    

    # Sidebar for input fields
    st.sidebar.header("🔍 Input Session Features")
    start_time = st.sidebar.slider("⏰ Start Time (Hour)", 0, 23, 8)
    end_time = st.sidebar.slider("⏰ End Time (Hour)", 0, 23, 10)
    charge_time = st.sidebar.number_input("🔋 Charge Time (Hours)", min_value=0.0, max_value=24.0, value=2.0)
    facility_type = st.sidebar.selectbox("🏢 Facility Type", options=["Public Charging Station", "Private Charging Station", "Corporate Charging Station", "Residential Charging Station" ], index=0)
    created_hour = st.sidebar.slider("🕒 Hour Created", 0, 23, 8)
    created_day = st.sidebar.slider("📅 Day Created", 1, 31, 15)
    created_month = st.sidebar.slider("📅 Month Created", 1, 12, 6)
    session_duration = st.sidebar.number_input("⏳ Session Duration (Hours)", min_value=0.0, max_value=24.0, value=2.5)
    kwh_per_hour = st.sidebar.number_input("⚡ kWh Per Hour", min_value=0.0, max_value=50.0, value=7.5)
    weekday = st.sidebar.selectbox("📆 Weekday", options=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], index=0)
    start_period = st.sidebar.selectbox("🌅 Start Period", options=["Morning", "Evening", "Night"], index=0)
    end_period = st.sidebar.selectbox("🌇 End Period", options=["Morning", "Evening", "Night"], index=0)

    # Collect inputs into a dictionary
    facility_type = list(facility_type_mapping.keys())[list(facility_type_mapping.values()).index(facility_type)]


    inputs = {
        "startTime": start_time,
        "endTime": end_time,
        "chargeTimeHrs": charge_time,
        "facilityType": facility_type,
        "created_hour": created_hour,
        "created_day": created_day,
        "created_month": created_month,
        "session_duration": session_duration,
        "kwh_per_hour": kwh_per_hour,
        "weekday": weekday,
        "startPeriod": start_period,
        "endPeriod": end_period,
    }

    if st.sidebar.button("🔮 Predict Energy Consumption"):
        features = preprocess_features(inputs)
        try:
            # Display message to prompt the user to scroll down
            st.info("🔽 Scroll down to see detailed predictions and visualizations!")

            # Prediction
            st.header("🔮 Prediction")

            prediction = model.predict(features)[0]
            st.success(f"✅ **Predicted Energy Consumption:** {prediction:.2f} kWh")

            # Visualization
            st.header("📊 Visualization")
            st.markdown("Here's a comparison between your input features and the predicted energy consumption.")

            # Bar Chart for comparison
            fig = go.Figure(data=[go.Bar(x=["Charge Time (hrs)", "Session Duration (hrs)", "Predicted kWh"],
                                        y=[inputs["chargeTimeHrs"], inputs["session_duration"], prediction],
                                        marker_color=["#1f77b4", "#ff7f0e", "#2ca02c"])])
            fig.update_layout(title="Prediction vs Inputs", yaxis_title="Values")
            st.plotly_chart(fig)

            # Load the original dataset
            df = pd.read_csv(DATA_PATH)

            # Apply inverse transform to unscale the data
            df[['session_duration', 'kwh_per_hour']] = scaler.inverse_transform(df[['session_duration', 'kwh_per_hour']])

            # Generate all visualizations
            create_visualizations(df)

        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")

if __name__ == "__main__":
    main()
