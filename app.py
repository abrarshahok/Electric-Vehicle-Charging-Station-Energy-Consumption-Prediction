import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

# Constants and Configuration
MODEL_PATH = "./models/xgb_model.pkl"
SCALER_PATH = "./scalers/scaler.pkl"
DATA_PATH = "./data/cleaned_station_data.csv"

# Load model and scaler
with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)
    
with open(SCALER_PATH, "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

def preprocess_features(inputs):
    """Preprocess features for prediction with full scaling"""
    # Create base features array
    base_features = np.array([
        inputs["chargeTimeHrs"],
        inputs["startTime"], 
        inputs["endTime"],
        inputs["created_day"],
        inputs["created_month"],
        inputs["kwh_per_hour"]
    ]).reshape(1, -1)
    
    # Scale all numerical features
    scaled_features = scaler.transform(base_features)
    
    # One-hot encodings
    weekday_cols = ["Mon", "Sat", "Sun", "Thu", "Tue", "Wed"]
    weekday_encoding = [1 if day == inputs["weekday"] else 0 for day in weekday_cols]
    
    period_cols = ["Evening", "Morning", "Night"]
    start_period_encoding = [1 if period == inputs["startPeriod"] else 0 for period in period_cols]
    end_period_encoding = [1 if period == inputs["endPeriod"] else 0 for period in period_cols]
    
    # Combine scaled features with categorical encodings
    return np.concatenate([
        scaled_features.flatten(),
        weekday_encoding,
        start_period_encoding,
        end_period_encoding
    ]).reshape(1, -1)

def collect_inputs():
    """Collect user inputs from sidebar"""
    return {
        "startTime": st.sidebar.slider("⏰ Start Time (Hour)", 0, 23, 8),
        "endTime": st.sidebar.slider("⏰ End Time (Hour)", 0, 23, 10),
        "chargeTimeHrs": st.sidebar.number_input("🔋 Charge Time (Hours)", 0.0, 24.0, 2.0),
        "created_day": st.sidebar.slider("📅 Day Created", 1, 31, 15),
        "created_month": st.sidebar.slider("📅 Month Created", 1, 12, 6),
        "kwh_per_hour": st.sidebar.number_input("⚡ kWh Per Hour", 0.0, 50.0, 7.5),
        "weekday": st.sidebar.selectbox("📆 Weekday", ["Mon", "Tue", "Wed", "Thu", "Sat", "Sun"]),
        "startPeriod": st.sidebar.selectbox("🌅 Start Period", ["Morning", "Evening", "Night"]),
        "endPeriod": st.sidebar.selectbox("🌇 End Period", ["Morning", "Evening", "Night"])
    }

def create_visualizations(df, prediction=None, inputs=None):
    """Create visualization dashboard with purple theme"""
    col1, col2 = st.columns(2)
    
    purple_color = '#9b4dca'  # Purple color for theme
    
    with col1:
        # Energy vs Duration
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['chargeTimeHrs'],
            y=df['kwhTotal'],
            mode='markers',
            name='Historical Energy Usage',
            marker=dict(color=purple_color)  # Purple for historical data points
        ))
        if prediction:
            fig.add_trace(go.Scatter(
                x=[inputs['chargeTimeHrs']],
                y=[prediction],
                mode='markers',
                name='Predicted Energy Usage',
                marker=dict(size=12, color='red')  # Keep red for predicted values
            ))
        fig.update_layout(
            title='Energy vs Duration',
            xaxis_title='Charge Duration (hours)',
            yaxis_title='Energy Usage (kWh)',
            font=dict(color='white')  # White font color
        )
        st.plotly_chart(fig)
        
        # Descriptive message
        st.markdown(
            "This chart shows the relationship between the charge duration and the energy usage "
            "(in kWh). The blue markers represent the historical data, while the red marker shows "
            "the predicted energy usage for the provided charge duration."
        )
    
    with col2:
        # Monthly trends
        monthly = df.groupby('created_month')['kwhTotal'].mean()
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=monthly.index, 
            y=monthly.values,
            name='Monthly Energy Usage',
            marker=dict(color=purple_color)  # Purple bars for monthly trends
        ))
        if prediction:
            fig.add_trace(go.Scatter(
                x=[inputs['created_month']],
                y=[prediction],
                mode='markers',
                name='Predicted Energy Usage',
                marker=dict(size=12, color='red')  # Keep red for predicted values
            ))
        fig.update_layout(
            title='Monthly Energy Consumption',
            xaxis_title='Month',
            yaxis_title='Average Energy Usage (kWh)',
            font=dict(color='white')  # White font color
        )
        st.plotly_chart(fig)
        
        # Descriptive message
        st.markdown(
            "This chart illustrates the average energy consumption per month. The purple bars represent "
            "historical monthly consumption, and the red marker indicates the predicted energy consumption "
            "for the selected month."
        )

def make_prediction(inputs):
    """Generate prediction"""
    try:
        features = preprocess_features(inputs)
        prediction = model.predict(features)[0]
        st.session_state.prediction = prediction
        st.session_state.inputs = inputs
        return True
    except Exception as e:
        st.error(f"Error: {e}")
        return False

def show_landing_page():
    """Display enhanced landing page"""
    # Header Section
    st.markdown("""
        <div style='text-align: center;'>
            <h1>🔋 EV Charging Station Energy Predictor</h1>
            <p style='font-size: 1.2em; color: #666;'>Smart Energy Prediction for Electric Vehicle Charging</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Key Features Section
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🎯 Predict")
        st.markdown("""
            - Accurate energy estimates
            - Real-time calculations
            - Smart predictions
        """)
    
    with col2:
        st.markdown("### 📊 Analyze")
        st.markdown("""
            - Usage patterns
            - Time-based trends
            - Energy consumption
        """)
    
    with col3:
        st.markdown("### 💡 Optimize")
        st.markdown("""
            - Cost efficiency
            - Usage insights
            - Performance metrics
        """)
    
    # Quick Start Guide
    st.markdown("---")
    st.markdown("### 🚀 Quick Start Guide")
    st.markdown("""
    1. Input charging parameters in the sidebar
    2. Click 'Predict Energy Consumption' to get energy consumption estimate
    3. View detailed analysis and comparisons
    """)
    
    # Sample Metrics
    st.markdown("---")
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    
    with metrics_col1:
        st.metric(label="Average Charging Time", value="2.5 hrs")
    with metrics_col2:
        st.metric(label="Typical Energy Usage", value="7.8 kWh")
    with metrics_col3:
        st.metric(label="Peak Hours", value="9AM-11AM")
    
    # Call to Action
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; padding: 2rem;'>
            <h3>Ready to Start?</h3>
            <p>Use the sidebar to input your charging parameters and get instant predictions!</p>
        </div>
    """, unsafe_allow_html=True)

def show_prediction_page():
    """Display prediction results"""
    st.title("Prediction Results")
    if hasattr(st.session_state, 'prediction'):
        st.success(f"✅ **Predicted Energy Consumption:**: {st.session_state.prediction:.2f} kWh")
        df = pd.read_csv(DATA_PATH)
        create_visualizations(df, st.session_state.prediction, st.session_state.inputs)

def main():
    st.set_page_config(page_title="EV Charging Predictor", layout="wide")
    
    if 'page' not in st.session_state:
        st.session_state.page = 'landing'

    st.sidebar.title("Input Features")
    inputs = collect_inputs()
    
    if st.sidebar.button("🔮 Predict Energy Consumption"):
        if make_prediction(inputs):
            st.session_state.page = 'prediction'
    
    if st.session_state.page == 'prediction':
        if st.sidebar.button("🏠 Back to Home"):
            st.session_state.page = 'landing'
    
    if st.session_state.page == 'landing':
        show_landing_page()
    else:
        show_prediction_page()

if __name__ == "__main__":
    main()
