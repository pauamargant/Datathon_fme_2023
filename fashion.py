from pytorch_metric_learning.distances import LpDistance
import matplotlib.pyplot as plt
import torch
import streamlit as st


def display_images(paths):
    fig = plt.figure(figsize=(10, 20))
    columns = 4
    rows = 5
    for i in range(1, min(columns * rows + 1, len(paths) + 1)):
        img = plt.imread(paths[i - 1][9:])
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
    plt.show()


def get_image_path(cod_modelo_color, outfit_data):
    return outfit_data[outfit_data["cod_modelo_color"] == cod_modelo_color][
        "des_filename"
    ].values[0][9:]


from pytorch_metric_learning.distances import LpDistance


def get_embedding(model, x):
    return model(torch.tensor(x.values, dtype=torch.float32, requires_grad=True))


def get_recommendation(model, x, item_id, k=10, embeddings=None):
    if embeddings is None:
        embeddings = get_embedding(model, x)

    Lp = LpDistance(normalize_embeddings=False)
    Lp_sim = Lp(embeddings, embeddings[item_id].unsqueeze(0))

    idx = torch.argsort(Lp_sim.squeeze())

    topk = list(zip(idx.tolist(), Lp_sim[idx].tolist()))

    topk.sort(key=lambda x: x[1], reverse=True)
    return topk[:k]


def find_item(filtered, k=1):
    # sample k rows from filtered
    return filtered.sample(k)["cod_modelo_color"].values
