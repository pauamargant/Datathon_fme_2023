import streamlit as st
import pickle

from fashion import *

import asyncio
import aiofiles
import pandas as pd
import torch


async def read_csv(file_path):
    async with aiofiles.open(file_path, mode="r") as file:
        content = await file.read()
        return pd.read_csv(pd.compat.StringIO(content))


@st.cache
async def load_data():
    loop = asyncio.get_event_loop()

    # Use asyncio.gather to concurrently read the CSV files
    X, product_data = await asyncio.gather(
        read_csv("dataset/X.csv"), read_csv("dataset/product_data_cleaned.csv")
    )

    # Load model.pkl using a separate thread or process
    model = await loop.run_in_executor(None, lambda: torch.load("model.pst"))

    return X, model, product_data


X, model, product_data = load_data()


st.title("Outfit Recommender")
st.write(
    "Welcome to the Outfit Recommender! This app will help you find the perfect outfit for any occasion. Simply select the category, color, and fabric of the outfit you are looking for and we will find the best outfit for you!"
)

st.sidebar.title("Outfit Feature Selection")

x = pd.read_csv("dataset/product_data.csv")

# Sidebar - Outfit Color Selection

# Sidebar select boxes with "All" as the default option
category = st.sidebar.selectbox(
    "Category", ["All"] + list(x["des_product_category"].unique()), index=0
)
color = st.sidebar.selectbox(
    "Color", ["All"] + list(x["des_agrup_color_eng"].unique()), index=0
)
fabric = st.sidebar.selectbox(
    "Fabric", ["All"] + list(x["des_fabric"].unique()), index=0
)
# Corrected button usage
if st.sidebar.button("Find Outfit", key="generate"):
    # run a function that will display the outfit
    # For now, let's just print a message
    st.write("Here is your outfit!")
    # textbox item_id
    item_id = st.text_input("Enter item_id", "0")
    recommendation = get_recommendation(model, x, item_id, k=10, embeddings=None)
    st.write(recommendation)
