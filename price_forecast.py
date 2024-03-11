import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from utils import data 
from datetime import datetime, timedelta
import plotly.graph_objects as go
import joblib
import numpy as np

##--------------------------------------- Functions ---------------------------------------##

def load_predictions():
    # Load predictions from CSV
    df = pd.read_csv('predicted_data.csv', parse_dates=True, index_col=0)
    return df

def plot_predictions(df, start_date, end_date):
    # Filter dataframe based on selected date range
    start_datetime = pd.to_datetime(start_date)
    end_datetime = pd.to_datetime(end_date)
    # Ensure the time component is set to start of the day for comparison
    start_datetime = pd.Timestamp(start_datetime.date())
    end_datetime = pd.Timestamp(end_datetime.date())
    mask = (df.index >= start_datetime) & (df.index <= end_datetime)
    filtered_df = df.loc[mask]

    # Create Plotly figure
    fig = go.Figure()

    # Assuming 'Middle', 'Upper', 'Lower' are columns in your dataframe
    # for predictions and 'Actual' for actual values (if available)
    fig.add_trace(go.Scatter(x=filtered_df.index, y=filtered_df['Middle'], mode='lines', name='Predicted Price'))
    fig.add_trace(go.Scatter(x=filtered_df.index, y=filtered_df['Upper'], fill='tonexty', mode='lines', name='Upper C.I.', line=dict(width=0)))
    fig.add_trace(go.Scatter(x=filtered_df.index, y=filtered_df['Lower'], fill='tonexty', mode='lines', name='Lower C.I.', line=dict(width=0), fillcolor='rgba(192,192,192,0.5)'))
    
    # Optional: If you have actual values to compare with
    if 'Actual' in filtered_df.columns:
        fig.add_trace(go.Scatter(x=filtered_df.index, y=filtered_df['Actual'], mode='lines+markers', name='Actual Price'))

    fig.update_layout(title='Energy Price Predictions', xaxis_title='Date', yaxis_title='Price', legend_title='Legend')
    return fig

def load_models():
    models = {}
    model_names = ['upper', 'middle', 'lower']
    
    for model_name in model_names:
        filename = f"{model_name}.pkl"
        model = joblib.load(filename)
        models[model_name] = model
    
    return models

def predict_energy_prices(model, df):
    # Placeholder for making predictions with your model
    predictions_middle = model['middle'].predict(df)
    predictions_upper = model['upper'].predict(df)
    predictions_lower = model['lower'].predict(df)
    
    predictions_df = pd.DataFrame({
        'Middle': predictions_middle,
        'Upper': predictions_upper,
        'Lower': predictions_lower
    })
    
    return predictions_df

# Function to generate synthetic dataset based on historical averages for the same day, month, and hour across years
def generate_synthetic_data_for_specific_hours(start_date, end_date, historical_df):
    date_range = pd.date_range(start=start_date, end=end_date, freq='H')
    synthetic_data = pd.DataFrame(index=date_range, columns=historical_df.columns)

    for column in synthetic_data.columns:
        synthetic_data[column] = synthetic_data[column].astype(float)
    
    # Select only numeric columns for which mean calculation makes sense
    numeric_cols = historical_df.select_dtypes(include=np.number).columns

    for single_date in date_range:
        matching_dates = historical_df[(historical_df.index.month == single_date.month) &
                                       (historical_df.index.day == single_date.day) &
                                       (historical_df.index.hour == single_date.hour)]
        for column in numeric_cols:  # Only iterate over numeric columns
            # Calculate mean for numeric columns
            synthetic_data.at[single_date, column] = matching_dates[column].mean()

    # Ensure all data in numeric columns is of float type
    synthetic_data[numeric_cols] = synthetic_data[numeric_cols].astype(float)

    return synthetic_data

# Function to add lag features to the DataFrame
def add_lag_features(df):
    df = df.sort_index()
    for column in df.columns:
        for lag in range(1, 3):  # Adjust the range if you need more/less lags
            df[f'{column}_lag_{lag}'] = df[column].shift(lag)
    df.dropna(inplace=True)  # Remove rows with NaN values resulting from shifting
    
    for column in df.columns:
        df[column] = df[column].astype('float64')
    return df

def prepare_and_adjust_predictions(start_date, end_date, predictions_df, yearly_pct_change, monthly_pct_change):
    # Generate the full date range
    full_date_range = pd.date_range(start=start_date, end=end_date, freq='H')
    
    # Adjust for the lag by removing the first 2 hours from the full_date_range
    adjusted_date_range = full_date_range[2:]
    
    # Ensure predictions_df has the correct length
    if len(predictions_df) != len(adjusted_date_range):
        raise ValueError("Length of predictions_df does not match the adjusted date range.")
    
    # Set the adjusted_date_range as the new index for the predictions DataFrame
    predictions_df.index = adjusted_date_range
    
    # Apply adjustments
    for index, row in predictions_df.iterrows():
        adjustment_factor = get_adjustment_factor(index, yearly_pct_change, monthly_pct_change)
        predictions_df.at[index, 'Middle'] *= adjustment_factor
        predictions_df.at[index, 'Upper'] *= adjustment_factor
        predictions_df.at[index, 'Lower'] *= adjustment_factor

    return predictions_df



def get_adjustment_factor(date, yearly_pct_change, monthly_pct_change):
    year = date.year
    year_month = pd.to_datetime(date).to_period('M')
    
    yearly_adjustment = yearly_pct_change.get(year, 0) / 100
    monthly_adjustment = monthly_pct_change.get(year_month, 0) / 100
    
    # Combine adjustments; this simplistic approach adds them, but you might want a more nuanced method
    adjustment_factor = 1 + ((yearly_adjustment + monthly_adjustment)/2)
    return adjustment_factor
'''
def apply_adjustments(predictions_df, yearly_pct_change, monthly_pct_change):
    # Adjust each prediction using the appropriate yearly and monthly changes
    for index, row in predictions_df.iterrows():
        adjustment_factor = get_adjustment_factor(index, yearly_pct_change, monthly_pct_change)
        
        predictions_df.at[index, 'Middle'] *= adjustment_factor
        predictions_df.at[index, 'Upper'] *= adjustment_factor
        predictions_df.at[index, 'Lower'] *= adjustment_factor
        
    return predictions_df

'''

def load_historical_data():
    historical_df = pd.read_csv('modeling_data.csv')
    historical_df['time'] = pd.to_datetime(historical_df['time'], utc=True)
    historical_df = historical_df.drop('price day ahead', axis=1)
    historical_df.set_index('time', inplace=True)
    historical_df = historical_df.sort_index()
    historical_df.index = historical_df.index.tz_localize(None)
    return historical_df

def calculate_price_change_percentages(modeling_df):
    # Add 'year' and 'month' columns based on the DataFrame's index
    modeling_df['year'] = modeling_df.index.year
    modeling_df['month'] = modeling_df.index.month

    # 1. Yearly price change calculation
    yearly_avg_price = modeling_df.groupby('year')['price actual'].mean()
    yearly_price_change_pct = yearly_avg_price.pct_change() * 100

    # 2. Monthly price change calculation
    modeling_df['year_month'] = modeling_df.index.to_period('M')
    monthly_avg_price = modeling_df.groupby('year_month')['price actual'].mean()
    monthly_price_change_pct = monthly_avg_price.pct_change(periods=12) * 100

    # Returning the calculated percentage changes
    return yearly_price_change_pct, monthly_price_change_pct

def run_prediction_process(start_date_str, end_date_str, historical_df, model):
    # Convert input strings to datetime objects
    start_date = pd.to_datetime(start_date_str)
    end_date = pd.to_datetime(end_date_str)
    
    # Generate the synthetic dataset
    synthetic_df = generate_synthetic_data_for_specific_hours(start_date, end_date, historical_df)
    synthetic_df.drop(columns=['year', 'month', 'year_month'], inplace=True)
    # Check if synthetic_df is empty or contains NaN values
    if synthetic_df.isnull().values.any():
        st.write("Synthetic data contains NaN values. Check if historical data for the given dates exists.")
        return
    
    # Add lag features
    synthetic_df_with_lags = add_lag_features(synthetic_df)
    
    synthetic_df_with_lags = synthetic_df_with_lags.drop('price actual', axis = 1)
    
    # Predict energy prices
    predictions_df = predict_energy_prices(model, synthetic_df_with_lags)
    
    return predictions_df

def aggregate_data(df, granularity):
    if granularity == 'Hourly':
        # No aggregation needed for hourly data
        agg_df = df.copy()
    elif granularity == 'Daily':
        agg_df = df.resample('D').mean()
    elif granularity == 'Weekly':
        agg_df = df.resample('W').mean()
    elif granularity == 'Monthly':
        agg_df = df.resample('M').mean()
    else:
        raise ValueError("Unsupported granularity")
    return agg_df

def plot_adjusted_predictions_with_granularity(df, granularity):

    # Aggregate the data based on the selected granularity
    agg_df = aggregate_data(df, granularity)

    # Plotting the aggregated data
    fig = plot_predictions(agg_df, agg_df.index.min(), agg_df.index.max())  # Adjusted to use the whole range of agg_df

    st.plotly_chart(fig)

def download_csv(df):
    # Convert DataFrame to CSV string
    csv = df.to_csv(index=True)  # Ensure index=True if you want to include datetime indices in the CSV
    
    # Create the download button
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='adjusted_predictions.csv',
        mime='text/csv',
    )


##--------------------------------------- Display Flow ---------------------------------------##

def show():
    st.title("Energy Price Prediction Visualization")
    
    df = load_predictions()  # Load your DataFrame with a DateTimeIndex
    
    # Convert pandas.Timestamp to datetime.date
    min_date, max_date = df.index.min().date(), df.index.max().date()
    
    # Calculate the end date for the first week
    one_week_later = min_date + timedelta(days=7)
    
    st.subheader("Select a date range to view predictions:")
    
    # Set the default value of the slider to the first week
    date_range = st.slider(
        "Select Date Range", 
        min_value=min_date, 
        max_value=max_date, 
        value=(min_date, one_week_later),  # Default to the first week
        format="MM/DD/YYYY"
    )
    
    start_date1, end_date1 = date_range
    
    # Ensure plot_predictions is compatible with datetime.date inputs
    fig1 = plot_predictions(df, start_date1, end_date1)
    st.plotly_chart(fig1)

    st.title("Energy Price Prediction and Adjustment Tool")

    # User inputs for data file and date range
    uploaded_file = None #st.file_uploader("Choose a CSV file", type=['csv'])
    start_date = st.date_input("Start date", datetime.today())
    end_date = st.date_input("End date", datetime.today())
    granularity = st.selectbox("Select Granularity", options=['Hourly', 'Daily', 'Weekly', 'Monthly'])

    if uploaded_file is None:
        historical_df = load_historical_data()
        #st.write(historical_df)
        yearly_pct_change, monthly_pct_change = calculate_price_change_percentages(historical_df)

        # Display the yearly and monthly percentage changes (optional)
        # st.write(yearly_pct_change)
        # st.write(monthly_pct_change)

        if st.button('Predict and Adjust Prices'):
            if end_date < start_date:
                st.error("End date must be after start date.")
            else:
                model = load_models()
                # Make sure to convert dates to strings if your function expects string inputs
                start_date_str = start_date.strftime('%Y-%m-%d')
                end_date_str = end_date.strftime('%Y-%m-%d')
                predictions_df = run_prediction_process(start_date_str, end_date_str, historical_df, model)
                #st.write(predictions_df)
                if predictions_df is not None:
                    # Note: prepare_and_adjust_predictions should replace apply_adjustments
                    # This assumes start_date and end_date are already appropriate datetime objects
                    adjusted_predictions_df = prepare_and_adjust_predictions(start_date, end_date, predictions_df, yearly_pct_change, monthly_pct_change)
                    #st.write("Adjusted Price Predictions", adjusted_predictions_df)
                    #fig = plot_price_predictions_with_intervals(adjusted_predictions_df)
                    #fig = plot_predictions(adjusted_predictions_df, start_date, end_date)
                    #st.plotly_chart(fig)
                    plot_adjusted_predictions_with_granularity(adjusted_predictions_df, granularity)
                    download_csv(adjusted_predictions_df)
                else:
                    st.error("No predictions could be made for the provided date range.")





	