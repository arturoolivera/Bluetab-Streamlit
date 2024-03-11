import streamlit as st
import home, project_overview, historical_analysis, price_forecast

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["🏠 Home", "🏭 Project Overview", "📈 Price Forecast",  "📊 Historical Analysis",])

    if page == "🏠 Home":
        home.show()
    elif page == "🏭 Project Overview":
        project_overview.show()
    elif page == "📊 Historical Analysis":
        historical_analysis.show()
    elif page == "📈 Price Forecast":
        price_forecast.show()

if __name__ == "__main__":
    main()