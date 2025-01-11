# Electric-Vehicle-Charging-Station-Energy-Consumption-Prediction

## 🚗 Overview

This project provides a tool to predict energy consumption at Electric Vehicle (EV) charging stations. With the rapid adoption of electric vehicles, optimizing energy usage at charging stations has become crucial for improving operational efficiency and ensuring customer satisfaction. This tool leverages predictive modeling to forecast energy consumption for charging sessions based on various input features.

## ⚡ Features

- **Energy Consumption Prediction**: Predict energy consumption (in kWh) for each charging session.
- **Data Visualization**: Visualize the relationship between input features and predicted energy consumption.
- **Customizable Inputs**: Adjust session details such as start time, duration, and facility type to get tailored predictions.
- **User-Friendly Interface**: Simple, clean, and intuitive interface built using Streamlit for ease of use.

## 🌍 Why This Tool is Needed

As the adoption of electric vehicles grows, the need for efficient management of energy consumption at charging stations becomes paramount. This tool helps charging station operators by:

- **Forecasting Energy Requirements**: Predict energy usage for each session.
- **Optimizing Resource Allocation**: Manage resources better, ensuring availability during peak hours.
- **Improving Customer Satisfaction**: Reduce long wait times and ensure consistent energy supply.

## 🛠️ How It Works

1. **Input Session Features**: Input details such as start time, charge duration, and facility type using the sidebar.
2. **Predict Energy Consumption**: Click the **Predict Energy Consumption** button to calculate the predicted energy usage.
3. **View Results**: See your prediction along with visualizations that help in understanding the energy consumption patterns.

## 🖥️ Requirements

- Python 3.x
- Streamlit
- Scikit-learn
- Plotly
- Pandas

You can install the required dependencies using `pip`:

```bash
pip install -r requirements.txt
```

## 🛠️ Setup

To run the app locally:

1. Clone this repository:

   ```bash
   git clone https://github.com/abrarshahok/Electric-Vehicle-Charging-Station-Energy-Consumption-Prediction.git
   ```

2. Navigate to the project directory:

   ```bash
   cd Electric-Vehicle-Charging-Station-Energy-Consumption-Prediction
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

5. Open the provided local URL in your browser to interact with the app.

## 📊 Visualizations

After submitting the session details, users can view the following:

- **Prediction vs Inputs Bar Chart**: A visual comparison between the predicted energy consumption and the provided inputs (e.g., charge time, session duration).
- **Additional Data Visualizations**: Insights into the dataset, helping to understand patterns in energy consumption based on historical data.

## 📄 Input Features

The following features can be customized:

- **Start Time**: The start time of the charging session (in hours).
- **End Time**: The end time of the charging session (in hours).
- **Charge Time (Hours)**: Duration for which the vehicle is charged.
- **Facility Type**: Type of charging station (e.g., Public, Private, Corporate, Residential).
- **Created Hour, Day, and Month**: Time-related features for when the charging session was created.
- **Session Duration (Hours)**: Total duration of the session.
- **kWh Per Hour**: Energy consumption rate (kWh per hour).
- **Weekday**: The day of the week the session takes place.
- **Start and End Period**: Period of the day when the session starts and ends (Morning, Evening, or Night).
