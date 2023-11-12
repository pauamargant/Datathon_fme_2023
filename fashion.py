from pytorch_metric_learning.distances import LpDistance
import matplotlib.pyplot as plt
import torch
import streamlit as st
import pickle
import pandas as pd


def display_images(paths):
    fig = plt.figure(figsize=(10, 20))
    columns = 4
    rows = 5
    for i in range(1, min(columns * rows + 1, len(paths) + 1)):
        img = plt.imread(paths[i - 1][9:])
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
    plt.show()


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


from pytorch_metric_learning.distances import LpDistance


def get_embedding(model, x):
    return model(torch.tensor(x.values, dtype=torch.float32, requires_grad=True))


@st.cache
def get_recommendation(model, x, item_id, k=10, embeddings=None):
    if embeddings is None:
        embeddings = get_embedding(model, x)

    Lp = LpDistance(normalize_embeddings=False)
    Lp_sim = Lp(embeddings, embeddings[item_id].unsqueeze(0))

    idx = torch.argsort(Lp_sim.squeeze())

    topk = list(zip(idx.tolist(), Lp_sim[idx].tolist()))

    topk.sort(key=lambda x: x[1], reverse=True)
    return topk[:k]


def get_model():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader

    class MyNN(nn.Module):
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

    x = pd.read_csv("x.csv")
    input_size = x.shape[1]
    hidden_size1 = 64
    hidden_size2 = 32
    output_size = 20

    model = MyNN(input_size, hidden_size1, hidden_size2, output_size)
    return model


model = get_model()

from restrictions import Outfit


def get_outfit(base, product_data):
    compatbile = False
    while not compatbile:
        # get random 3 other items
        other_items = product_data.sample(3)
        # to list of rows
        other_items = [row for _, row in other_items.iterrows()]
        # make outfit
        outfit = Outfit([base] + other_items)
        # check if outfit is
        if outfit.get_error() == 0:
            compatbile = True
    return outfit
