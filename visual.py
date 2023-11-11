import streamlit as st
import pandas as pd

# put emotes

st.title("Outfit Recommender")
st.write("Welcome to the Outfit Recommender! This app will help you find the perfect outfit for any occasion. Simply select the category, color, and fabric of the outfit you are looking for and we will find the best outfit for you!")

st.sidebar.title("Outfit Feature Selection")

x = pd.read_csv('dataset/product_data.csv')

# Sidebar - Outfit Color Selection

category = st.sidebar.selectbox("Category", x["des_product_category"].unique())
color = st.sidebar.selectbox("Color", x["des_agrup_color_eng"].unique())
fabric = st.sidebar.selectbox("Fabric", x["des_fabric"].unique())
   
 