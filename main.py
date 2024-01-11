#!/usr/bin/env python3
from time import sleep, time
from matplotlib.offsetbox import AnchoredText
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import sample, shuffle
from scipy.sparse import data
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from minisom import MiniSom
from xarray.core.dataarray import DataArray
from itertools import product
from random import randint
from multiprocessing import Queue, Process
from pickle import dump, load
from parse_be_csv import get_bes

IMAGE_PATH = "/mnt/c/Users/Julian/Desktop/test.png"
# IMAGE_PATH = "test.png"
COORDS = None

MODEL_FILE = "model.som"
DATASET = "resampled.nc"
TRAINING_STEPS = "ani1/epoche_{:04}.png"
TRAINING_STEPS = "/mnt/d/som/ani1/epoche_{:02}_step_{:06}.png"
ERROR_FILE = "/mnt/d/som/ani1-{}-error.png"
HEATMAP_TOTAL = "/mnt/d/som/heatmap-complete-dataset.png"
HEATMAP_BE = "/mnt/d/som/heatmap-be.png"
HEATMAP_RECENTLY = "/mnt/d/som/heatmap-2017-2022.png"
FINAL_MODEL = "/mnt/d/som/final-model-weights.png"
FIG_FAC = 2
FIGSIZE = [6.4 * FIG_FAC, 4.8 * FIG_FAC]
EPOCHS = 2
STEP_ERROR_PLOT_INTERVAL = 5  # 1 to plot every epoche
STEP_MODEL_FRAME_INTERVAL = 50
RENDERING_PROCESSES = 2


def plot_errors(filename, title, y_axis_label, errors):
    x_values = [i * STEP_ERROR_PLOT_INTERVAL for i in range(len(errors))]
    fig = plt.figure(887788, figsize=FIGSIZE)
    fig.clf()
    fig, ax = plt.subplots()
    ax.plot(x_values, errors)
    ax.set(xlabel="Sample count", ylabel=y_axis_label, title=title)
    ax.grid()
    fig.savefig(filename)


def plot_heat(sample_counts):
    fig = plt.figure(np.random.randint(0, 9999999999), figsize=FIGSIZE)
    average = np.average(sample_counts)
    minimum = np.min(sample_counts)
    maximum = np.max(sample_counts)
    plt.imshow(sample_counts, cmap="hot", interpolation="nearest", vmin=minimum - 20, vmax=maximum + 20)
    for i in range(sample_counts.shape[0]):
        for j in range(sample_counts.shape[1]):
            color = "white"
            if sample_counts[i, j] > average:
                color = "black"
            plt.text(
                j,
                i,
                sample_counts[i, j],
                ha="center",
                va="center",
                color=color,
                fontsize=20,
            )
    plt.colorbar()
    ax = plt.gca()
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)


def plotting(da: xr.Dataset, values):
    fig = plt.figure(6723489)  # No figsize
    fig.clf()
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent(COORDS, crs=ccrs.PlateCarree())
    ax.stock_img()
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.COASTLINE)
    ax.coastlines(resolution="50m")
    x = da.coords["latitude"].values
    y = da.coords["longitude"].values
    pcm = ax.pcolormesh(
        y,
        x,
        values,
        vmin=-2.5,
        vmax=2.5,
        shading="nearest",
        cmap="jet",
    )
    cb = plt.colorbar(pcm, pad=0.05, label="Abweichung (100m)", location="bottom", shrink=0.5)
    # cb.set_label("Abweichung (m)", size=20)


def raise_for_nan_inf(ar: np.array):
    if np.isnan(ar).any():
        raise ValueError("Contains NaN")
    if not np.isfinite(ar).all():
        raise ValueError("Contains Inf")


def model_plot(model, coordinates, mark=None):
    model = np.array(model)
    fig = plt.figure(345435, figsize=FIGSIZE)
    fig.clf()
    SPACE = 0.1
    gs = fig.add_gridspec(
        *model.shape[:2],
        wspace=SPACE,
        hspace=SPACE,
        left=SPACE / 2,
        right=1 - SPACE / 2,
        bottom=SPACE / 2,
        top=1 - SPACE / 2,
    )

    # print(np.max(model), np.min(model))
    for i, j in product(range(model.shape[0]), range(model.shape[1])):
        ax = fig.add_subplot(gs[i, j], projection=ccrs.PlateCarree())
        if (i, j) == mark:
            print("Marker")
            ax.set_facecolor("orange")
        ax.set_extent(COORDS, crs=ccrs.PlateCarree())
        ax.stock_img()
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.COASTLINE)
        ax.coastlines(resolution="50m")

        x = coordinates["latitude"].values
        y = coordinates["longitude"].values

        x = coordinates["latitude"].values
        y = coordinates["longitude"].values
        m = model[i, j].reshape((x.shape[0], y.shape[0]))
        ax.pcolormesh(
            y,
            x,
            m,
            vmin=-2.5,
            vmax=2.5,
            shading="nearest",
            cmap="jet",
        )
    return fig


def rendering_thread(q: Queue, coordinates):
    while True:
        qv = q.get()
        if qv is None:
            return
        (filename, values) = qv
        start = time()
        fig = model_plot(values, coordinates)
        fig.savefig(filename)
        duration = time() - start
        short = filename.split("/")[-1]
        short = ".".join(short.split(".")[:-1])
        print(f" [{short}-{duration:.2f}s] ", end="", flush=True)


def open_dataset():
    global COORDS
    dataset = xr.open_dataset(DATASET)
    COORDS = [
        dataset.coords["longitude"][0],
        dataset.coords["longitude"][-1],
        dataset.coords["latitude"][0],
        dataset.coords["latitude"][-1],
    ]
    return dataset

def decay_function(learning_rate, t, max_iter):
    original_learning_rate = learning_rate / (1+t/(max_iter/2))
    return original_learning_rate ** 1.2

def main():
    dataset = open_dataset()
    model = MiniSom(
        4, 4, len(dataset.coords["latitude"]) * len(dataset.coords["longitude"]),
        sigma=0.6,
        decay_function=decay_function,
    )
    dataset_np = np.array(dataset["z"].as_numpy())
    print(dataset_np.shape)

    dataset_np = dataset_np.reshape(
        (dataset_np.shape[0], dataset_np.shape[1] * dataset_np.shape[2])
    )
    print(dataset_np.shape)
    dataset_filtered = [
        layer
        for layer in dataset_np
        if (not np.isnan(layer).any() and np.isfinite(layer).all())
    ]
    print("Size filtered:", len(dataset_filtered))
    print("Size original:", len(dataset_np))

    dataset_np = np.array(dataset_filtered)
    print("Normalize")
    average = np.average(dataset_np, axis=0)
    for layer in dataset_np:
        layer -= average
    dataset_np /= 1000
    print("Max:", np.max(dataset_np), "Min:", np.min(dataset_np))
    print("Train")
    # for point in dataset_np[:4]:
    #     if np.isnan(point).any():
    #         print("Nan")
    #     if not np.isfinite(point).all():
    #         print("Inf")

    # # Maunal training
    # model.train_random(dataset_np, num_iteration=9)
    q = Queue(maxsize=RENDERING_PROCESSES * 2)

    rendering_processes = [
        Process(target=rendering_thread, args=(q, dataset.coords))
        for _ in range(RENDERING_PROCESSES)
    ]
    for p in rendering_processes:
        "Comment"
        p.start()

    print("Init PCA")
    model.pca_weights_init(dataset_np[:500])

    dataset_size = dataset_np.shape[0]
    start = time()
    quant_errors = []
    topographic_errors = []
    # for epoche in []:  # range(EPOCHS):
    for epoche in range(EPOCHS):
        samples = list(range(dataset_size))
        shuffle(samples)
        for i in range(len(samples)):
            index = samples[i]
            step = epoche * dataset_size + i
            sample = dataset_np[index]
            raise_for_nan_inf(sample)
            model.update(sample, model.winner(sample), step, EPOCHS * dataset_size)
            if i % STEP_ERROR_PLOT_INTERVAL == 0:
                quantization_error = model.quantization_error(dataset_np)
                topographic_error = model.topographic_error(dataset_np)
                normalized_quantization_error = quantization_error / (sample.shape[0]) * 100
                quant_errors.append(normalized_quantization_error)
                topographic_errors.append(topographic_error)
                duration = time() - start
                print()
                print(f"Epoche {epoche}, step {i}, {duration:.2f}s. ", end="")
                print(
                    f"Quantization (normalized): {normalized_quantization_error}; Topographic: {topographic_error}",
                    end="",
                    flush=True,
                )
            if i % STEP_MODEL_FRAME_INTERVAL == 0:
                filename = TRAINING_STEPS.format(epoche, i)
                q.put((filename, model.get_weights()))
            # if i == 30:
            #     break
    print("Plot final weights")
    fig = model_plot(model.get_weights(), dataset.coords)
    fig.savefig(FINAL_MODEL)
    print("Plot errors")
    plot_errors(ERROR_FILE.format("quant"), "Quantization Error (normalized)", "Average Deviation (m)", quant_errors)
    plot_errors(ERROR_FILE.format("quant-100"), "Quantization Error (Skip 100)", "Average Deviation (m)", quant_errors[100:])
    plot_errors(ERROR_FILE.format("topo"), "Topographic Error", "Error", topographic_errors)
    plot_errors(ERROR_FILE.format("topo-100"), "Topographic Error (Skip 100)", "Error", topographic_errors[100:])
    # with open(MODEL_FILE, "rb") as f:
    #     model = load(f)
    with open(MODEL_FILE, "wb") as f:
        dump(model.get_weights(), f)

    # Total heatmap
    print("Plot", HEATMAP_TOTAL)
    (width, height, _) = model.get_weights().shape
    count_heat = np.zeros((width, height), dtype=np.int32)
    for sample in dataset_np:
        (x, y) = model.winner(sample.flatten())
        count_heat[x, y] += 1
    plot_heat(count_heat)
    plt.savefig(HEATMAP_TOTAL)

    # Blocking Event Heatmap
    print("Plot", HEATMAP_BE)
    (width, height, _) = model.get_weights().shape
    count_heat = np.zeros((width, height), dtype=np.int32)
    for be in get_bes():
        try:
            day = dataset.sel(time=be)
        except KeyError:
            # print("Warning, date", be, "not found in dataset")
            continue
        day_np = day["z"].to_numpy().flatten()
        day_np -= average
        day_np /= 1000
        (x, y) = model.winner(day_np)
        count_heat[x, y] += 1
    plot_heat(count_heat)
    plt.savefig(HEATMAP_BE)

    # Heatmap of recent dates
    print("Plot", HEATMAP_RECENTLY)
    (width, height, _) = model.get_weights().shape
    count_heat = np.zeros((width, height), dtype=np.int32)
    recent_years = dataset.sel(time=slice("2017-01-01", "2023-01-01"))
    recent_years_np = recent_years["z"].as_numpy()
    for sample in recent_years_np:
        sample = sample.to_numpy().flatten()
        if np.isnan(sample).any():
            continue
        sample -= average
        sample /= 1000
        (x, y) = model.winner(sample)
        count_heat[x, y] += 1
    plot_heat(count_heat)
    plt.savefig(HEATMAP_RECENTLY)

    for _ in rendering_processes:
        # Signal processes to terminate
        q.put(None)
    for p in rendering_processes:
        p.join()
    return


if __name__ == "__main__":
    main()
