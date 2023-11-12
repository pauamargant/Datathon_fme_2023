import streamlit as st
import pandas as pd
from torch import nn
import pickle
import torch
from streamlit_image_select import image_select

from fashion import *


# Add a button to trigger the popup
@st.cache_data
def load_data():
    X = pd.read_csv("dataset/X.csv")
    product_data = pd.read_csv("dataset/product_data_cleaned.csv")
    # load model.pkl
    model = torch.load("model.pth")
    return X, model, product_data


def show_image(id):
    outfit_data = pd.read_csv("dataset/product_data_cleaned.csv")

    # with open("dataset/dict_index_modelo.pickle", "rb") as f:
    #     dict_index_modelo = pickle.load(f)
    p = get_image_path(id, outfit_data)
    # remove "datathon/" from path
    st.write(p)
    # show image on path p
    st.image(p)

    # load dict_index_modelo from pickle


product_data, model, product_data = load_data()


st.title("Outfit Recommender")
st.write(
    "Welcome to the Outfit Recommender! This app will help you find the perfect outfit for any occasion. Simply select the category, color, and fabric of the outfit you are looking for and we will find the best outfit for you!"
)

st.sidebar.title("Outfit Feature Selection")


# Sidebar - Outfit Color Selection

# Sidebar select boxes with "All" as the default option
category = st.sidebar.selectbox(
    "Category", ["All"] + list(product_data["des_product_category"].unique()), index=0
)
if category != "All":
    filtered = product_data[product_data["des_product_category"] == category]
else:
    filtered = product_data
# add sidebar for des_product_family
family = st.sidebar.selectbox(
    "Family", ["All"] + list(filtered["des_product_family"].unique()), index=0
)
if family != "All":
    filtered = filtered[filtered["des_product_family"] == family]
color = st.sidebar.selectbox(
    "Color", ["All"] + list(filtered["des_agrup_color_eng"].unique()), index=0
)
if color != "All":
    filtered = filtered[filtered["des_agrup_color_eng"] == color]
fabric = st.sidebar.selectbox(
    "Fabric", ["All"] + list(filtered["des_fabric"].unique()), index=0
)
outfit = []

# Corrected button usage
# if st.sidebar.button("Find Outfit", key="generate"):
for _ in range(5):
    # run a function that will display the outfit
    # For now, let's just print a message
    st.write("Here is your outfit!")
    # textbox item_id
    # ask for input and store in item_id

    codes = find_item(filtered, min(len(filtered), 10))
    img = image_select(
        "Label",
        [get_image_path(code, product_data) for code in codes],
        return_value="index",
        use_container_width=False,
    )
    outfit.append(get_image_path(codes[img], product_data))
    # wait while not clicked button continue


# Display the selected outfit images
for i, img_path in enumerate(outfit):
    st.image(img_path, caption=f"Item {i + 1}", use_column_width=True)
