#!/usr/bin/env python3
from matplotlib.pyplot import hot
from numpy.random import choice
from main import *

FILENAME_SAMPLE = "/mnt/d/som/update/sample.png"
FILENAME_MODEL_BEFORE = "/mnt/d/som/update/model_before.png"
FILENAME_MODELE_AFTER = "/mnt/d/som/update/model_after.png"
BMU_FILE = "/mnt/d/som/update/bmu.txt"

AVERAGE = None

def init_normalize(full_dataset_np):
    global AVERAGE
    AVERAGE = np.average(full_dataset_np, axis=0)

def normalize(data):
    data -= AVERAGE
    data /= 1000
    return data

def get_dataset_np(dataset):
    dataset_np = np.array(dataset["z"].as_numpy())
    # dataset_np = dataset_np.reshape(
    #     (dataset_np.shape[0], dataset_np.shape[1] * dataset_np.shape[2])
    # )
    dataset_filtered = [
        layer
        for layer in dataset_np
        if (not np.isnan(layer).any() and np.isfinite(layer).all())
    ]
    dataset_np = np.array(dataset_filtered)
    return dataset_np

def get_model_pretrained():
    dataset = open_dataset()
    model = MiniSom(
        4, 4, len(dataset.coords["latitude"]) * len(dataset.coords["longitude"])
    )
    dataset_np = get_dataset_np(dataset)
    dataset_filtered = [
        layer
        for layer in dataset_np
        if (not np.isnan(layer).any() and np.isfinite(layer).all())
    ]
    dataset_np = np.array(dataset_filtered)
    for layer in dataset_np:
        layer = normalize(layer)
    dataset_np = dataset_np.reshape(
        (dataset_np.shape[0], dataset_np.shape[1] * dataset_np.shape[2])
    )
    model.pca_weights_init(dataset_np[:500])
    indexes = list(range(dataset_np.shape[0]))
    shuffle(indexes)
    for i in range(500):
        index = indexes[i]
        sample = dataset_np[index]
        model.update(sample, model.winner(sample), i, 1000)
    return model


def main():
    dataset = open_dataset()
    init_normalize(get_dataset_np(dataset))
    # Hottest day in BW, Germany
    day = dataset.sel(time="2022-07-20")
    hot_day = day["z"]
    hot_day_numpy = hot_day.to_numpy()
    model = get_model_pretrained()

    plotting(dataset, (hot_day_numpy - AVERAGE) / 1000 )
    plt.savefig(FILENAME_SAMPLE)

    model_plot(model.get_weights(), dataset.coords)
    plt.savefig(FILENAME_MODEL_BEFORE)

    sample = hot_day_numpy.flatten()
    sample -= AVERAGE.flatten()
    sample /= 1000
    bmu = model.winner(sample)
    with open(BMU_FILE, mode="w") as f:
        f.write(str(bmu))
    model.update(sample, bmu, 5, 100)
    model_plot(model.get_weights(), dataset.coords, mark=bmu)
    plt.savefig(FILENAME_MODELE_AFTER)



if __name__ == '__main__':
    main()
