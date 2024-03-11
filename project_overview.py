import streamlit as st
from utils import data  # Ensure this module is correctly implemented

def show():
    st.header('Project Overview')

    st.markdown("""
        ### Understanding the Spanish Energy Market 

        Spain is at the forefront of the energy transition due to its energy and climate change policies. The current Spanish framework for energy and climate is based on the 2050 objectives of national climate neutrality, 100% renewable energy in the electricity mix, and 97% renewable energy in the total energy mix. As such, it is centred on the massive development of renewable energy, particularly solar, wind, and renewable hydrogen, increasing energy efficiency and improving electrification. This is an opportunity for the country to not only stimulate the economy by creating jobs through the modernisation of industry but also to support vulnerable populations, improve energy security, and support RD&D and innovation.
        
        ### Key Figures of the Spanish Market in 2018
        
        - **Electrical Energy Demand in Spain:** 268,877 GWh
        - **Renewable Energy Generation in the Peninsular System:** 40.1%
        - **Installed Power Capacity in Spain:** 104,094 MW
        - **Percentage of Renewable Energy:** 46.7%
        - **Wind Energy - Second Source of Peninsular Electricity Generation:** 19.8%
        
        ### Datasets Overview
        - **Energy Dataset:** A comprehensive dataset encompassing detailed hourly data on various energy generation sources, including biomass, fossil fuels, nuclear, and renewables, alongside consumption patterns and market prices.
        - **Weather Features Dataset:** Detailed weather data capturing temperature, pressure, humidity, wind speed, and weather conditions across multiple Spanish cities. This dataset plays a crucial role in understanding the climatological impacts on energy demand and renewable energy production.

        ### Project Goals
        - **Explore and Analyze the Provided Datasets:** To conduct a thorough analysis of the energy and weather datasets to uncover patterns, trends, and insights.
        - **Develop a Robust Energy Price Forecasting Model:** Leveraging historical data to predict future energy prices, accounting for various factors including demand, supply, and weather conditions.
        - **Propose an Analytical and Predictive Model with Real-world Applications:** To create actionable insights and tools that can aid in optimizing energy production, informing policy, and supporting decision-making processes within the Spanish energy sector.

        ### Significance
        This project aims to bridge the gap between historical data analysis and future forecasting, providing stakeholders with the tools they need to navigate the complexities of the energy market. By integrating data analysis with predictive modeling, we strive to contribute to a more sustainable and efficient energy future.
    """)

    st.markdown("""
        ### Next Steps
        - **Data Exploration:** A deep dive into the energy and weather datasets to understand the data structure and quality.
        - **Preliminary Analysis:** Identifying correlations and patterns that can influence energy prices.
        - **Model Development:** Building and validating predictive models for energy price forecasting.
        - **Visualization and Dashboard Creation:** Developing interactive visualizations and dashboards for an intuitive understanding of the data and models.
    """)
