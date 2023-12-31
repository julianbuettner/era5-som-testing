from time import sleep, time
from matplotlib.offsetbox import AnchoredText
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import shuffle
from scipy.sparse import data
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from minisom import MiniSom
from xarray.core.dataarray import DataArray
from itertools import product
from random import randint
from multiprocessing import Queue, Process

IMAGE_PATH = "/mnt/c/Users/Julian/Desktop/test.png"
IMAGE_PATH = "test.png"
COORDS = None

DATASET = "resampled.nc"
TRAINING_STEPS = "ani1/epoche_{:04}.png"
TRAINING_STEPS = "/mnt/d/som/ani1/epoche_{:02}_step_{:06}.png"
ERROR_FILE = "/mnt/d/som/ani1-{}-error.png"
FIG_FAC = 2
FIGSIZE = [6.4 * FIG_FAC, 4.8 * FIG_FAC]
STEPS = 90 * 1000
EPOCHS = 2
STEP_PLOT_INTERVAL = 5  # 1 to plot every epoche
RENDERING_PROCESSES = 10


def plot_errors(filename, title, errors):
    fig = plt.figure(887788, figsize=FIGSIZE)
    fig.clf()
    fig, ax = plt.subplots()
    ax.plot(errors)
    ax.set(xlabel="Sample count", ylabel=title, title=title)
    ax.grid()
    fig.savefig(filename)


def plotting(da: xr.Dataset, values):
    fig = plt.figure(6723489, figsize=FIGSIZE)
    fig.clf()
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent(COORDS, crs=ccrs.PlateCarree())
    ax.stock_img()
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.COASTLINE)
    ax.coastlines(resolution="50m")
    x = da.coords["latitude"].values
    y = da.coords["longitude"].values
    ax.pcolormesh(
        y,
        x,
        values,
        vmin=-2.5,
        vmax=2.5,
        shading="nearest",
    )


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
            ax.set_facecolor('orange')
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
        short = filename.split('/')[-1]
        short = '.'.join(short.split('.')[:-1])
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


def main():
    dataset = open_dataset()
    model = MiniSom(
        4, 4, len(dataset.coords["latitude"]) * len(dataset.coords["longitude"])
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
        p.start()



    print("Init PCA")
    model.pca_weights_init(dataset_np[:500])

    dataset_size = dataset_np.shape[0]
    start = time()
    quant_errors = []
    topographic_errors = []
    for epoche in []: #range(EPOCHS):
        samples = list(range(dataset_size))
        shuffle(samples)
        for i in range(len(samples)):
            index = samples[i]
            step = epoche * dataset_size + i
            sample = dataset_np[index]
            raise_for_nan_inf(sample)
            model.update(sample, model.winner(sample), step, EPOCHS * dataset_size)
            if i % STEP_PLOT_INTERVAL == 0:
                quantization_error = model.quantization_error(dataset_np)
                topographic_error = model.topographic_error(dataset_np)
                quant_errors.append(quantization_error)
                topographic_errors.append(topographic_error)
                filename = TRAINING_STEPS.format(epoche, i)
                # q.put((filename, model.get_weights()))
                duration = time() - start
                print()
                print(f"Epoche {epoche}, step {i}, {duration:.2f}s. ", end="")
                print(
                    f"Quantization: {quantization_error}; Topographic: {topographic_error}",
                    end="", flush=True
                )
    plot_errors(ERROR_FILE.format("quant"), "Quantization Error", quant_errors)
    plot_errors(ERROR_FILE.format("quant-100"), "Quantization Error (Skip 100)", quant_errors[100:])
    plot_errors(ERROR_FILE.format("topo"), "Topographic Error", topographic_errors)
    plot_errors(ERROR_FILE.format("topo-100"), "Topographic Error (SKip 100)", topographic_errors[100:])
    for _ in rendering_processes:
        # Signal processes to terminate
        q.put(None)
    for p in rendering_processes:
        p.join()
    return

    

if __name__ == "__main__":
    main()
