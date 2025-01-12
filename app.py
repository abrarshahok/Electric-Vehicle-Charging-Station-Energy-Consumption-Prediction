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

def create_visualizations(df, prediction=None, inputs=None):
    """Visualization with optional prediction overlay"""
    
    st.header("Time-Based Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        hourly = df.groupby('created_hour')['kwh_per_hour'].mean().reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hourly['created_hour'], 
                               y=hourly['kwh_per_hour'],
                               mode='lines+markers',
                               name='Historical',
                               line=dict(color='#1f77b4')))
        if prediction and inputs:
            fig.add_trace(go.Scatter(x=[inputs['created_hour']],
                                   y=[prediction/inputs['chargeTimeHrs']],
                                   mode='markers',
                                   name='Predicted',
                                   marker=dict(color='#e74c3c', size=12)))
            fig.update_layout(
                title={
                    'text': 'Average Hourly Energy Usage (kWh)',
                    'y': 0.95,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                },
                xaxis_title='Hour of Day',
                yaxis_title='Average Energy Consumption (kWh/hour)'
            )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        monthly = df.groupby('created_month')['kwh_per_hour'].mean().reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=monthly['created_month'],
                               y=monthly['kwh_per_hour'],
                               mode='lines+markers',
                               name='Historical',
                               line=dict(color='#1f77b4')))
        if prediction and inputs:
            fig.add_trace(go.Scatter(x=[inputs['created_month']],
                                   y=[prediction/inputs['chargeTimeHrs']],
                                   mode='markers',
                                   name='Predicted',
                                   marker=dict(color='#e74c3c', size=12)))
            fig.update_layout(
                title={
                    'text': 'Average Monthly Energy Usage (kWh)',
                    'y': 0.95,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                },
                xaxis_title='Month of Year',
                yaxis_title='Average Energy Consumption (kWh/month)'
            )
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    
    with col3:
        weekdays = ['Mon', 'Tue', 'Wed', 'Thu', 'Sat', 'Sun']
        weekday_counts = [df[f'weekday_{day}'].sum() for day in weekdays]
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Bar(
            name='Historical',
            x=weekdays,
            y=weekday_counts,
            marker_color='#1f77b4'
        ))
        
        # Prediction overlay
        if prediction and inputs:
            pred_counts = [prediction if day == inputs['weekday'] else 0 for day in weekdays]
            fig.add_trace(go.Bar(
                name='Predicted',
                x=weekdays,
                y=pred_counts,
                marker_color='#e74c3c'
            ))
            
        fig.update_layout(
            title='Weekday Usage Comparison',
            barmode='group',
            yaxis_title='Number of Sessions'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        periods = ['Morning', 'Evening', 'Night']
        period_counts = [df[f'startPeriod_{period}'].sum() for period in periods]
        
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Bar(
            name='Historical',
            x=periods,
            y=period_counts,
            marker_color='#1f77b4'
        ))
        
        # Prediction overlay
        if prediction and inputs:
            pred_period = [prediction if period == inputs['startPeriod'] else 0 for period in periods]
            fig.add_trace(go.Bar(
                name='Predicted',
                x=periods,
                y=pred_period,
                marker_color='#e74c3c'
            ))
            
        fig.update_layout(
            title='Time Period Distribution',
            barmode='stack',
            yaxis_title='Energy Consumption (kWh)'
        )
        st.plotly_chart(fig, use_container_width=True)

    st.header("Usage Patterns")
    col5, col6 = st.columns(2)

    with col5:
        fig = go.Figure(data=go.Histogram(x=df['chargeTimeHrs'],
                                       nbinsx=30,
                                       name='Historical',
                                       marker_color='#1f77b4'))
        if prediction and inputs:
            fig.add_vline(x=inputs['chargeTimeHrs'], 
                         line_dash="dash",
                         line_color="#e74c3c",
                         annotation_text="Predicted")
        fig.update_layout(title='Charging Duration Distribution')
        st.plotly_chart(fig, use_container_width=True)

    with col6:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['chargeTimeHrs'],
                               y=df['kwhTotal'],
                               mode='markers',
                               name='Historical',
                               marker=dict(color='#1f77b4', size=8)))
        if prediction and inputs:
            fig.add_trace(go.Scatter(x=[inputs['chargeTimeHrs']],
                                   y=[prediction],
                                   mode='markers',
                                   name='Predicted',
                                   marker=dict(color='#e74c3c', size=12)))
        fig.update_layout(title='Energy vs Duration')
        st.plotly_chart(fig, use_container_width=True)

def show_landing_page():
    st.title("🔋 EV Charging Station Energy Consumption Predictor")
    st.markdown("""
    **Welcome to the EV Charging Station Predictor!**
    A powerful tool to optimize energy usage and enhance operational efficiency for EV charging stations.
    """)
    
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
    
    st.header("✨ Key Features")
    st.markdown("""
    - **Energy Consumption Prediction**: Get accurate kWh predictions for each charging session.
    - **Data Visualization**: Understand energy trends through interactive charts.
    - **Customizable Inputs**: Tailor predictions to specific scenarios with detailed input options.
    - **User-Friendly Design**: Easy-to-use interface with clear results and actionable insights.
    """)
    
    st.header("🔍 How It Works")
    st.markdown("""
    1. **Open sidebar** and input session details such as start time, charge duration, and facility type.
    2. Click on **Predict Energy Consumption**.
    3. Get an instant prediction along with visual insights to optimize energy usage.
    """)

def collect_inputs():
    """Collect and validate user inputs from sidebar"""
    inputs = {
        "startTime": st.sidebar.slider("⏰ Start Time (Hour)", 0, 23, 8),
        "endTime": st.sidebar.slider("⏰ End Time (Hour)", 0, 23, 10),
        "chargeTimeHrs": st.sidebar.number_input("🔋 Charge Time (Hours)", min_value=0.0, max_value=24.0, value=2.0),
        "created_hour": st.sidebar.slider("🕒 Hour Created", 0, 23, 8),
        "created_day": st.sidebar.slider("📅 Day Created", 1, 31, 15),
        "created_month": st.sidebar.slider("📅 Month Created", 1, 12, 6),
        "session_duration": st.sidebar.number_input("⏳ Session Duration (Hours)", min_value=0.0, max_value=24.0, value=2.5),
        "kwh_per_hour": st.sidebar.number_input("⚡ kWh Per Hour", min_value=0.0, max_value=50.0, value=7.5),
        "weekday": st.sidebar.selectbox("📆 Weekday", options=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], index=0),
        "startPeriod": st.sidebar.selectbox("🌅 Start Period", options=["Morning", "Evening", "Night"], index=0),
        "endPeriod": st.sidebar.selectbox("🌇 End Period", options=["Morning", "Evening", "Night"], index=0)
    }
    
    facility = st.sidebar.selectbox("🏢 Facility Type", options=list(facility_type_mapping.values()), index=0)
    inputs["facilityType"] = list(facility_type_mapping.keys())[list(facility_type_mapping.values()).index(facility)]
    
    return inputs

def make_prediction(inputs):
    """Process inputs and generate prediction"""
    try:
        features = preprocess_features(inputs)
        prediction = model.predict(features)[0]
        st.session_state.prediction = prediction
        st.session_state.inputs = inputs
        return True
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
        return False

def show_prediction_page():
    """Display prediction results and visualizations"""
    st.title("🔮 Prediction Dashboard")
    
    if hasattr(st.session_state, 'prediction'):
        st.success(f"✅ **Predicted Energy Consumption:** {st.session_state.prediction:.2f} kWh")
        
        df = pd.read_csv(DATA_PATH)
        df[['session_duration', 'kwh_per_hour']] = scaler.inverse_transform(
            df[['session_duration', 'kwh_per_hour']]
        )
        
        create_visualizations(df, 
                            prediction=st.session_state.prediction,
                            inputs=st.session_state.inputs)

def main():
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = 'landing'

    # Page configuration
    st.set_page_config(
        page_title="EV Charging Station Predictor",
        page_icon="🔋",
        layout="wide"
    )
    
    # Sidebar setup
    st.sidebar.header("🔍 Input Session Features")
    inputs = collect_inputs()
    
    # Navigation
    if st.sidebar.button("🔮 Predict Energy Consumption"):
        if make_prediction(inputs):
            st.session_state.page = 'prediction'
            st.rerun()
            
    if st.session_state.page == 'prediction':
        if st.sidebar.button("🏠 Back to Home"):
            st.session_state.page = 'landing'
            st.rerun()
    
    # Display current page
    if st.session_state.page == 'landing':
        show_landing_page()
    else:
        show_prediction_page()

if __name__ == "__main__":
    main()
