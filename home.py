import streamlit as st

def show():
    st.title("Welcome to the Energy Price Forecasting App")

    # Introduction Text
    st.write("""
    This application provides insights into energy price forecasting, using advanced machine learning models to predict future prices based on historical data and market conditions. 
    """)

    # Additional Information
    st.write("""
    ## App Sections
    Navigate through the application using the sidebar to explore different sections, including:
    
    - **Project Overview:** A detailed description of the project's background, objectives, and methodologies.
    - **Price Forecast:** Interactive forecasts of energy prices using the latest data and models.
    - **Historical Analysis:** Insights into historical price trends and the factors influencing energy prices.
    
    ## Collaboration
    This project represents a partnership between IE University and Bluetab, combining academic research and industry expertise to tackle real-world challenges in the energy sector.
    """)

       # Display Logos
    col1, col2 = st.columns([1, 1])  # Adjust the column weights if needed

    with col1:
        st.image("ie_logo.webp", width=200)  # Adjust the width as needed

    with col2:
        st.write("&#160;", unsafe_allow_html=True)  
        st.write("&#160;", unsafe_allow_html=True)  
        st.image("bluetab_logo.png", width=200)  # Adjust the width as needed

    st.write("Use the sidebar to navigate through the application.")

if __name__ == "__main__":
    show()
