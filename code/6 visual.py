import streamlit as st
import pandas as pd
from torch import nn
import pickle
import torch

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
    outfit_data = pd.read_csv("dataset/outfit_data.csv")

    def get_image_path(cod_modelo_color, outfit_data):
        return outfit_data[outfit_data["cod_modelo_color"] == cod_modelo_color][
            "des_filename"
        ].values[0]

    with open("datasets/dict_index_modelo.pkl", "rb") as f:
        dict_index_modelo = pickle.load(f)
    p = get_image_path(dict_index_modelo[id], outfit_data)
    st.write(p)
    # show image on path p
    st.image(p)

    # load dict_index_modelo from pickle


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
    # recommendation = get_recommendation(model, x, item_id, k=10, embeddings=None)
    # st.write(recommendation)
    # button
    st.write("dasasasdsa")
    show_image(item_id)
    if st.button("Show Outfit"):
        st.write("sdasd")
