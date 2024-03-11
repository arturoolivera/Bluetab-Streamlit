import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px


def load_data():
    data_path = 'merged_data_streamlit.csv'
    df = pd.read_csv(data_path)
    data = df.copy()
    data['time'] = pd.to_datetime(df['time'])
    data.columns = data.columns.str.replace(' ', '_')

data = load_data()