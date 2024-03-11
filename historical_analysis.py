import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
# from utils import data
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# ------------------------------------------------ FUNCTION DEFINITIONS: ------------------------------------------------

# Price Evolution Over Time:

def price_evolution_time_graph_h(df):

    fig = px.line(df, x='time', y=['price_actual', 'price_day_ahead'])

    fig = px.line(df, x='time', y=['price_actual', 'price_day_ahead'])

    # Update line colors and initially set visibility
    for i, trace in enumerate(fig.data):
        # Update line color
        if trace.name in display_colors:
            fig.data[i].update(line=dict(color=display_colors[trace.name]))
        
        # Set 'price_day_ahead' series initially hidden
        if trace.name == 'price_day_ahead':
            fig.data[i].visible = 'legendonly'

    # Update legend labels
    for column in ['price_actual', 'price_day_ahead']:
        fig.for_each_trace(lambda t: t.update(name=column.replace('_', ' ').title()) if t.name == column else ())

    # Update axis labels, hover labels, and layout
    fig.update_layout(
        xaxis_title='Time',
        yaxis_title='Price ($)',
        title='Price Evolution Over Time',
        legend_title_text=None
    )

    # Customize hovertemplate based on granularity
    if granularity == 'H':
        hover_template = '%{x|%B %d, %Y, %H:%M}<br>€%{y:.2f} / MWh<extra></extra>'
    elif granularity == 'D':
        hover_template = '%{x|%B %d, %Y}<br>€%{y:.2f} / MWh<extra></extra>'
    elif granularity == 'W':
        hover_template = '%{x|%B %d, %Y}<br>€%{y:.2f} / MWh<extra></extra>'
    elif granularity == 'M':
        hover_template = '%{x|%B, %Y}<br>€%{y:.2f} / MWh<extra></extra>'
    else:
        hover_template = '%{x}<br>€%{y:.2f} / MWh<extra></extra>'  # Default

    fig.update_traces(hovertemplate=hover_template)

    # Display the figure in Streamlit
    st.plotly_chart(fig)
    pass

# Energy Consumption vs Energy Production Over Time:

def energy_production_vs_energy_load_time_graph_h(df):

    fig = px.line(df, x='time', y=['total_load_actual', 'total_energy_production'])

    # Update line colors using display_colors dictionary
    for column, color in display_colors.items():
        fig.update_traces(line=dict(color=color), selector=dict(name=column))

    # Update legend labels
    for column in ['total_load_actual', 'total_energy_production']:
        fig.update_traces(name=column.replace('_', ' ').title(), selector=dict(name=column))

    # Update axis labels
    fig.update_layout(xaxis_title='Time', yaxis_title='Energy (MWh)')

    # Customize hovertemplate based on granularity
    if granularity == 'H':
        hover_template = '%{x|%B %d, %Y, %H:%M}<br>%{y} MWh<extra></extra>'
    elif granularity == 'D':
        hover_template = '%{x|%B %d, %Y}<br>%{y} MWh<extra></extra>'
    elif granularity == 'W':
        hover_template = '%{x|%B %d, %Y}<br>%{y} MWh<extra></extra>'
    elif granularity == 'M':
        hover_template = '%{x|%B, %Y}<br>%{y} MWh<extra></extra>'
    else:
        hover_template = '%{x}<br>%{y} MWh<extra></extra>'  # Default

    # Update hover labels
    fig.update_traces(hovertemplate=hover_template)  # Rounded to 1 decimal place

    # Update title
    fig.update_layout(title='Energy Consumption vs Energy Production Over Time')

    # Remove the legend title (assuming you no longer want the 'Series:' text)
    fig.update_layout(legend_title_text=None)

    # Display the figure in Streamlit
    st.plotly_chart(fig)
    pass

# Production Share Per Source:

def production_share_pie_chart_h(df, grouping):

    # Define custom colors for each section using display_colors dictionary
    custom_colors = [display_colors.get(col, '#999999') for col in df.columns if col.startswith('generation') or col == 'Total Minor Sources']

    # Determine which columns to plot based on the grouping parameter
    if grouping == 'Clean vs. Non-Clean':
        columns_to_plot = ['clean_energy', 'non_clean_energy']
    elif grouping == 'Renewable vs. Non-Renewable':
        columns_to_plot = ['renewable_energy', 'non_renewable_energy']
    elif grouping == 'Intermittent vs. Non-Intermittent':
        columns_to_plot = ['intermittent_energy', 'non_intermittent_energy']
    else:  # If grouping is 'None', display all generation columns
        columns_to_plot = [col for col in df.columns if col.startswith('generation')]

    # Sum the selected columns
    generation_totals = df[columns_to_plot].sum()

    # Calculate percentages for the pie chart slices
    percentages = 100 * generation_totals / generation_totals.sum()
    threshold = 5  # Define your threshold here, e.g., 5%

    # Group categories below threshold into 'Total Minor Sources'
    if grouping == 'None':
        below_threshold_categories = percentages < threshold
        if below_threshold_categories.any():
            others_value = generation_totals[below_threshold_categories].sum()
            generation_totals = generation_totals[~below_threshold_categories]
            generation_totals['Total Minor Sources'] = others_value
            percentages = 100 * generation_totals / generation_totals.sum()
            custom_colors = [display_colors.get(col, '#999999') for col in generation_totals.index if col.startswith('generation') or col == 'Total Minor Sources']

    # Format series names as per the new specifications
    formatted_names = [' '.join(col.replace('generation_', '').split('_')).capitalize() for col in generation_totals.index]
    text_labels = [f"{percent:.2f}%" if percent >= threshold else '' for name, percent in zip(formatted_names, percentages)]

    # Create a pie chart
    fig = go.Figure(data=[
        go.Pie(
            labels=formatted_names,
            values=generation_totals.values,
            hovertemplate='%{label}: %{value} MWh (%{percent})<extra></extra>',
            text=text_labels,
            textinfo='text',
            marker=dict(colors=custom_colors),
            insidetextorientation='radial'
        )
    ])

    # Update layout to customize title and text
    fig.update_layout(
        title='Production Share Per Energy Source',
        font=dict(size=12, color='black'),
        legend=dict(font=dict(size=10, color='black'), title=dict(text='Energy Source')),
    )
    
    # Display the figure in Streamlit
    st.plotly_chart(fig)
    pass

def production_share_evolution_time_graph_h(df, grouping):
    
    # Determine which columns to plot based on the grouping parameter
    if grouping == 'Clean vs. Non-Clean':
        columns_to_plot = ['clean_energy', 'non_clean_energy']
        legend_names = ['Clean Energy', 'Non Clean Energy']  # Adjusted for readability
    elif grouping == 'Renewable vs. Non-Renewable':
        columns_to_plot = ['renewable_energy', 'non_renewable_energy']
        legend_names = ['Renewable Energy', 'Non Renewable Energy']
    elif grouping == 'Intermittent vs. Non-Intermittent':
        columns_to_plot = ['intermittent_energy', 'non_intermittent_energy']
        legend_names = ['Intermittent Energy', 'Non Intermittent Energy']
    else:
        # If grouping is 'None', initially select all generation columns
        columns_to_plot = [col for col in df.columns if col.startswith('generation_')]

    # Check for columns with total = 0 and filter them out
    column_totals = df[columns_to_plot].sum()
    columns_to_include = column_totals[column_totals != 0].index.tolist()
    columns_to_plot = [col for col in columns_to_plot if col in columns_to_include]

    # Adjust legend names to match the filtered columns
    if grouping == 'None':
        legend_names = [col.replace('generation_', '').replace('_', ' ').capitalize() for col in columns_to_plot]
    else:
        # For predefined groupings, ensure the legend names list matches the filtered columns
        legend_names = [name for name, col in zip(legend_names, columns_to_plot) if col in columns_to_include]

    # Create a new color mapping based on the formatted names and filtered columns
    new_color_mapping = {name: display_colors[original_name] for name, original_name in zip(legend_names, columns_to_plot) if original_name in display_colors}

    # Prepare the data for 100% stacked area chart
    df_melted = df.melt(id_vars=['time'], value_vars=columns_to_plot, var_name='Energy Type', value_name='Production')

    # Calculate the total production for each time point to normalize the data for 100% stacking
    total_production_at_each_point = df_melted.groupby('time')['Production'].transform('sum')
    df_melted['Percentage'] = (df_melted['Production'] / total_production_at_each_point) * 100

    # Apply the formatted names for the melted dataframe's 'Energy Type' for correct legend labels
    name_mapping = {original: formatted for original, formatted in zip(columns_to_plot, legend_names)}
    df_melted['Energy Type'] = df_melted['Energy Type'].map(name_mapping)

    # Plotting the 100% stacked area chart
    fig = px.area(df_melted, x='time', y='Percentage', color='Energy Type',
                  title='Production Share Evolution Over Time',
                  labels={'Percentage': 'Share (%)'},
                  groupnorm='percent',
                  color_discrete_map=new_color_mapping)  # Use the new color mapping here

    fig.update_layout(yaxis_ticksuffix='%', xaxis_title='Time', yaxis_title='Share of Total Production')
    fig.update_traces(hoverinfo='x+y+name', hovertemplate='%{y:.2f}%')

    # Display the figure in Streamlit
    st.plotly_chart(fig)

    pass

def transform_data(data, start_date, end_date, granularity):
    
     # Step 1: Filter data by date range
    filtered_data = data[(data['time'] >= start_date) & (data['time'] <= end_date)].copy()

    # Step 2: Define mapping of aggregation functions for each column
    aggregation_functions = {
        'generation_biomass': 'sum',
        'generation_fossil_brown_coal/lignite': 'sum',
        'generation_fossil_coal_derived_gas': 'sum',
        'generation_fossil_gas': 'sum',
        'generation_fossil_hard_coal': 'sum',
        'generation_fossil_oil': 'sum',
        'generation_fossil_oil_shale': 'sum',
        'generation_fossil_peat': 'sum',
        'generation_geothermal': 'sum',
        'generation_hydro_pumped_storage_consumption': 'sum',
        'generation_hydro_run_of_river_and_poundage': 'sum',
        'generation_hydro_water_reservoir': 'sum',
        'generation_marine': 'sum',
        'generation_nuclear': 'sum',
        'generation_other': 'sum',
        'generation_other_renewable': 'sum',
        'generation_solar': 'sum',
        'generation_waste': 'sum',
        'generation_wind_offshore': 'sum',
        'generation_wind_onshore': 'sum',
        'forecast_solar_day_ahead': 'sum',
        'forecast_wind_offshore_eday_ahead': 'sum',
        'forecast_wind_onshore_day_ahead': 'sum',
        'total_load_forecast': 'sum',
        'total_load_actual': 'sum',
        'price_day_ahead': 'mean',
        'price_actual': 'mean',
        'temp_Valencia': 'mean',
        'pressure_Valencia': 'mean',
        'humidity_Valencia': 'mean',
        'wind_speed_Valencia': 'mean',
        'wind_deg_Valencia': 'mean',
        'rain_1h_Valencia': 'sum',
        'rain_3h_Valencia': 'sum',
        'snow_3h_Valencia': 'sum',
        'clouds_all_Valencia': 'mean',
        'delta_T_Valencia': 'mean',
        'temp_Madrid': 'mean',
        'pressure_Madrid': 'mean',
        'humidity_Madrid': 'mean',
        'wind_speed_Madrid': 'mean',
        'wind_deg_Madrid': 'mean',
        'rain_1h_Madrid': 'sum',
        'rain_3h_Madrid': 'sum',
        'snow_3h_Madrid': 'sum',
        'clouds_all_Madrid': 'mean',
        'delta_T_Madrid': 'mean',
        'temp_Bilbao': 'mean',
        'pressure_Bilbao': 'mean',
        'humidity_Bilbao': 'mean',
        'wind_speed_Bilbao': 'mean',
        'wind_deg_Bilbao': 'mean',
        'rain_1h_Bilbao': 'sum',
        'rain_3h_Bilbao': 'sum',
        'snow_3h_Bilbao': 'sum',
        'clouds_all_Bilbao': 'mean',
        'delta_T_Bilbao': 'mean',
        'temp_Seville': 'mean',
        'pressure_Seville': 'mean',
        'humidity_Seville': 'mean',
        'wind_speed_Seville': 'mean',
        'wind_deg_Seville': 'mean',
        'rain_1h_Seville': 'sum',
        'rain_3h_Seville': 'sum',
        'snow_3h_Seville': 'sum',
        'clouds_all_Seville': 'mean',
        'delta_T_Seville': 'mean',
        'temp_Barcelona': 'mean',
        'pressure_Barcelona': 'mean',
        'humidity_Barcelona': 'mean',
        'wind_speed_Barcelona': 'mean',
        'wind_deg_Barcelona': 'mean',
        'rain_1h_Barcelona': 'sum',
        'rain_3h_Barcelona': 'sum',
        'snow_3h_Barcelona': 'sum',
        'clouds_all_Barcelona': 'mean',
        'delta_T_Barcelona': 'mean',
        'intermittent_energy': 'sum',
        'non_intermittent_energy': 'sum',
        'renewable_energy': 'sum',
        'non_renewable_energy': 'sum',
        'clean_energy': 'sum',
        'non_clean_energy': 'sum',
        'total_energy_production': 'sum'
    }

    # Step 3: Group and aggregate data according to granularity
    transformed_data = filtered_data.groupby(pd.Grouper(key='time', freq=granularity)).agg(aggregation_functions).reset_index()
    
    return transformed_data

# Generate a list of monthly options from Jan 2014 to Dec 2018
def generate_monthly_dates(start_year, start_month, end_year, end_month):
    start_date = datetime(start_year, start_month, 1)
    end_date = datetime(end_year, end_month, 1)
    date_list = [start_date]
    while date_list[-1] < end_date:
        next_month = date_list[-1].replace(day=28) + timedelta(days=4)  # this will never fail
        date_list.append(next_month - timedelta(days=next_month.day - 1))
    return date_list

date_options = generate_monthly_dates(2014, 1, 2018, 12)

# Create a dictionary to map user-friendly options to values
granularity_options = {
    'Hourly': 'H',
    'Daily': 'D',
    'Weekly': 'W',
    'Monthly': 'M'
}

# Create a dictionary to map user-friendly options to values
grouping_options = {
    'None': 'None',
    'Clean vs. Non-Clean': 'Clean vs. Non-Clean',
    'Renewable vs. Non-Renewable': 'Renewable vs. Non-Renewable',
    'Intermittent vs. Non-Intermittent': 'Intermittent vs. Non-Intermittent'
}

# ------------------------------------------------ IMPORT DATA: ------------------------------------------------

data_path = 'full_energy_streamlit.csv'
df = pd.read_csv(data_path)
data = df.copy()
data['time'] = pd.to_datetime(df['time'])

# ------------------------------------------------ INITIALIZE VARIABLES TO DEFAULT VALUES: ------------------------------------------------

start_date = '2014-01-01'
end_date = '2018-12-31'
granularity = 'W'  # Possible values: 'H' | 'D' | 'W' | 'M' 
grouping = 'None' # Possible values: 'None' | 'Clean vs. Non-Clean' | 'Renewable vs. Non-Renewable' | 'Intermittent vs. Non-Intermittent'

display_colors = {
    'generation_biomass': '#999900',
    'generation_fossil_brown_coal/lignite': '#99004c',
    'generation_fossil_coal_derived_gas': '#b266ff',
    'generation_fossil_gas': '#9933ff',
    'generation_fossil_hard_coal': '#7f00ff',
    'generation_fossil_oil': '#6600cc',
    'generation_fossil_oil_shale': '#4c0099',
    'generation_fossil_peat': '#330066',
    'generation_geothermal': '#bcbd22',
    'generation_hydro_pumped_storage_aggregated': '#004c99',
    'generation_hydro_pumped_storage_consumption': '#0066cc',
    'generation_hydro_run_of_river_and_poundage': '#0080ff',
    'generation_hydro_water_reservoir': '#3399ff',
    'generation_marine': '#0000ff',
    'generation_nuclear': '#b2ff66',
    'generation_other': '#ff6666',
    'generation_other_renewable': '#66ffb2',
    'generation_solar': '#ffde00',
    'generation_waste': '#994c00',
    'generation_wind_offshore': '#00cccc',
    'generation_wind_onshore': '#99ffff',
    'forecast_solar_day_ahead': '#ff7f0e',
    'forecast_wind_offshore_eday_ahead': '#00cccc',
    'forecast_wind_onshore_day_ahead': '#99ffff',
    'total_load_forecast': '#a0a0a0',
    'total_load_actual': '#9a00ff',
    'price_day_ahead': '#a0a0a0',
    'price_actual': '#7f00ff',
    'intermittent_energy': '#ffde00',
    'non_intermittent_energy': '#c0c0c0',
    'renewable_energy': '#009900',
    'non_renewable_energy': '#c0c0c0',
    'clean_energy': '#0080FF',
    'non_clean_energy': '#c0c0c0',
    'total_energy_production': '#ff007f',
    'Total Minor Sources': '#606060'
}

# ------------------------------------------------ DISPLAY FLOW: ------------------------------------------------

def show():
    st.title("Historical Behaviour of Energy Price")

    # Data Range Selector
    st.sidebar.header("Existing Data Range:")
    start_date, end_date = st.sidebar.select_slider(
        "Select a date range",
        options=[date.strftime('%B %Y') for date in date_options],
        value=(date_options[0].strftime('%B %Y'), date_options[-1].strftime('%B %Y'))
    )

    # Date Range Granularity Selector
    granularity_label = st.sidebar.selectbox(
        "Select date range granularity",
        ['Hourly', 'Daily', 'Weekly', 'Monthly'],
        index=1  # Set the default index to 1 for 'Daily'
    )
    
    # Get the value for granularity from the dictionary
    granularity = granularity_options[granularity_label]
    
    transformed_data = transform_data(data, start_date, end_date, granularity)

    # Grouping Selector
    grouping_label = st.sidebar.selectbox(
        "Select grouping",
        ['None', 'Clean vs. Non-Clean', 'Renewable vs. Non-Renewable', 'Intermittent vs. Non-Intermittent']
    )

    # Get the value for granularity from the dictionary
    grouping = grouping_options[grouping_label]

    # Price Evolution Time Graph
    st.markdown("### Price Evolution Over Time")
    price_evolution_time_graph_h(transformed_data)

    # Energy Production vs Energy Load Time Graph
    st.markdown("### Energy Production vs Energy Load Over Time")
    energy_production_vs_energy_load_time_graph_h(transformed_data)

    # Production Share Per Source Pie Chart
    st.markdown("### Production Share Per Source")
    production_share_pie_chart_h(transformed_data, grouping)

    # Production Share Per Source Evolution Time Graph
    st.markdown("### Production Share Per Source Evolution Over Time")
    production_share_evolution_time_graph_h(transformed_data, grouping)


# ------------------------------------------------ END ------------------------------------------------