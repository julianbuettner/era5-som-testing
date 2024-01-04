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
FIG_FAC = 2
FIGSIZE = [6.4 * FIG_FAC, 4.8 * FIG_FAC]
STEPS = 90 * 1000
EPOCHS = 3
STEP_PLOT_INTERVAL = 5  # 1 to plot every epoche
RENDERING_PROCESSES = 10


def plotting(ax, da: xr.DataArray, index=0):
    x = da.coords["latitude"].values
    y = da.coords["longitude"].values
    ax.pcolormesh(
        y,
        x,
        da.values[index],
        shading="nearest",
    )


def raise_for_nan_inf(ar: np.array):
    if np.isnan(ar).any():
        raise ValueError("Contains NaN")
    if not np.isfinite(ar).all():
        raise ValueError("Contains Inf")

def model_plot(model, coordinates):
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
        

def main():
    global COORDS
    dataset = xr.open_dataset(DATASET)
    COORDS = [
        dataset.coords["longitude"][0],
        dataset.coords["longitude"][-1],
        dataset.coords["latitude"][0],
        dataset.coords["latitude"][-1],
    ]

    # fig = plt.figure(0)
    # ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    # ax.set_extent(COORDS, crs=ccrs.PlateCarree())
    # ax.add_feature(cfeature.LAND)
    # ax.add_feature(cfeature.COASTLINE)
    # ax.coastlines(resolution="50m")
    #
    # sample = geopot.as_numpy().reshape(
    #     (
    #         data_dim[0],
    #         data_dim[1] * data_dim[2],
    #     )
    # )
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
    for epoche in range(EPOCHS):
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
                filename = TRAINING_STEPS.format(epoche, i)
                q.put((filename, model.get_weights()))
                duration = time() - start
                print()
                print(f"Epoche {epoche}, step {i}, {duration:.2f}s. ", end="")
                print(
                    f"Quantization: {quantization_error}; Topographic: {topographic_error}",
                    end="", flush=True
                )
    for _ in rendering_processes:
        # Signal processes to terminate
        q.put(None)
    for p in rendering_processes:
        p.join()
    return

    for i in range(STEPS):
        choice = np.random.choice(dataset["time"])
        sample = dataset["z"].sel(time=choice)
        sample_np = np.array(sample.to_numpy().flat)
        model.update(sample_np, model.winner(sample_np), i, STEPS)
        quantization_error = model.quantization_error(dataset_np)
        if i % 1000 == 0:
            print("Step {}/{}".format(i, STEPS))
            fig = model_plot(model.get_weights(), dataset)
            fig.savefig(TRAINING_STEPS.format(i))
    return

    plt.figure(0)
    for i in range(300):
        plotting(ax, geopot, index=i)
        plt.savefig(IMAGE_PATH)
        sleep(1)
        return


if __name__ == "__main__":
    main()
