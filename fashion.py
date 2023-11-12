from pytorch_metric_learning.distances import LpDistance
from pytorch_metric_learning.distances import LpDistance
import matplotlib.pyplot as plt
import torch
import streamlit as st
import pickle
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class MyNN(nn.Module):
    """
    A neural network with three fully connected layers and ReLU activation functions.

    Args:
        input_size (int): The size of the input layer.
        hidden_size1 (int): The size of the first hidden layer.
        hidden_size2 (int): The size of the second hidden layer.
        output_size (int): The size of the output layer.
    """

    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        return x


def display_images(paths):
    """
    Display images from the given list of file paths.

    Args:
        paths (list): A list of file paths.

    Returns:
        None
    """
    fig = plt.figure(figsize=(10, 20))
    columns = 4
    rows = 5
    for i in range(1, min(columns * rows + 1, len(paths) + 1)):
        img = plt.imread(paths[i - 1][9:])
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
    plt.show()


import pandas as pd
import pickle


def get_image_path(
    cod_modelo_color: str, outfit_data: pd.DataFrame, id: bool = False
) -> str:
    """
    Given a cod_modelo_color, returns the corresponding image path from the outfit_data DataFrame.

    Parameters:
    cod_modelo_color (str): The cod_modelo_color to search for.
    outfit_data (pd.DataFrame): The DataFrame containing the outfit data.
    id (bool): If True, cod_modelo_color is assumed to be an index and is converted to its corresponding value.

    Returns:
    str: The image path corresponding to the given cod_modelo_color.
    """
    if id:
        index_dict = pickle.load(open("dataset/dict_index_modelo.pickle", "rb"))
        cod_modelo_color = index_dict[cod_modelo_color]
    return outfit_data[outfit_data["cod_modelo_color"] == cod_modelo_color][
        "des_filename"
    ].values[0]


def get_image_path(cod_modelo_color, outfit_data, id=False):
    if id:
        index_dict = pickle.load(open("dataset/dict_index_modelo.pickle", "rb"))
        cod_modelo_color = index_dict[cod_modelo_color]
    return outfit_data[outfit_data["cod_modelo_color"] == cod_modelo_color][
        "des_filename"
    ].values[0]


def find_item(filtered, k=1):
    # sample k rows from filtered
    return filtered.sample(k)["cod_modelo_color"].values


def get_embedding(model, x):
    # remove first column
    x_s = x.iloc[:, 1:]
    return model(torch.tensor(x_s.values, dtype=torch.float32, requires_grad=True))


def get_recommendation(model, x, item_id, k=10, embeddings=None):
    if embeddings is None:
        embeddings = get_embedding(model, x)
    # load dataset/dict_modelo_index.pickle
    index_dict = pickle.load(open("dataset/dict_modelo_index.pickle", "rb"))
    item_id = index_dict[item_id]
    Lp = LpDistance(normalize_embeddings=True)

    Lp_sim = Lp(embeddings)
    # get row i
    Lp_sim = Lp_sim[:, item_id]
    idx = torch.argsort(Lp_sim.squeeze())

    topk = list(zip(idx.tolist(), Lp_sim[idx].tolist()))

    topk.sort(key=lambda x: x[1], reverse=True)
    print(topk[:5])
    return topk[:k]


def create_outfit(base, rows):
    """
    Given a base item and a list of rows, returns a list of cod_modelo_color values that make up an outfit.

    Args:
        base (pd.DataFrame): The base item.
        rows (list): A list of rows.

    Returns:
        list: A list of cod_modelo_color values.
    """

    outfit = [base]
    for row in rows:
        outfit.append(row)
    return outfit


@st.cache_data
def get_model():
    x = pd.read_csv("x.csv")
    input_size = x.shape[1] - 1
    hidden_size1 = 64
    hidden_size2 = 32
    output_size = 20

    model = MyNN(input_size, hidden_size1, hidden_size2, output_size)
    return model


model = get_model()
