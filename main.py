from time import sleep
from matplotlib.offsetbox import AnchoredText
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from minisom import MiniSom
from xarray.core.dataarray import DataArray
from itertools import product
from random import randint

IMAGE_PATH = "/mnt/c/Users/Julian/Desktop/test.png"
COORDS = None

DATASET = "full500.nc"
# DATASET = "test500.nc"
TRAINING_STEPS = "/mnt/d/som/ani1/training_{:05}.png"
FIG_FAC = 2
FIGSIZE = [6.4 * FIG_FAC, 4.8 * FIG_FAC]
STEPS = 90 * 1000
BATCHSIZE = 70

def plotting(ax, da: xr.DataArray, index=0):
    x = da.coords["latitude"].values
    y = da.coords["longitude"].values
    ax.pcolormesh(
        y,
        x,
        da.values[index],
        shading="nearest",
    )


def model_plot(model, da: xr.DataArray):
    model = np.array(model)
    fig = plt.figure(345435, figsize=FIGSIZE)
    SPACE = 0.1
    gs = fig.add_gridspec(
        *model.shape[:2],
        wspace=SPACE,
        hspace=SPACE,
        left=SPACE / 2,
        right=1 - SPACE / 2,
        bottom=SPACE / 2,
        top=1 - SPACE / 2
    )

    for i, j in product(range(model.shape[0]), range(model.shape[1])):
        ax = fig.add_subplot(gs[i, j], projection=ccrs.PlateCarree())
        ax.set_extent(COORDS, crs=ccrs.PlateCarree())
        ax.stock_img()
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.COASTLINE)
        ax.coastlines(resolution="50m")

        x = da.coords["latitude"].values
        y = da.coords["longitude"].values

        x = da.coords["latitude"].values
        y = da.coords["longitude"].values
        m = model[i, j].reshape((x.shape[0], y.shape[0]))
        ax.pcolormesh(
            y,
            x,
            m,
            vmin=50,
            vmax=60,
            shading="nearest",
            cmap='jet',
        )
    return fig


def resample(a: xr.Dataset):
    a = a.resample(time="1D").mean()
    a = a.coarsen(longitude=4, boundary="trim").mean()
    a = a.coarsen(latitude=4, boundary="trim").mean()
    a = a / 1000
    return a

def load_random_selection_of_data(size=1000):
    data = xr.open_dataset(DATASET)
    time_length = len(data["time"])
    time_start = randint(0, time_length - BATCHSIZE - 1)
    time_slice = slice(data["time"][time_start], data["time"][time_start + BATCHSIZE])
    print("Time slice starting from", data["time"][time_start])
    dataset = data.sel(time=time_slice)
    return resample(dataset)


def main():
    global COORDS
    dataset = load_random_selection_of_data()
    # print(dataset["z"])
    # return
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
    model = MiniSom(4, 4, len(dataset.coords["latitude"]) * len(dataset.coords["longitude"]))
    # model.pca_weights_init(sample[:4])
    print("Train")

    for i in range(STEPS):
        choice = np.random.choice(dataset["time"])
        sample = dataset["z"].sel(time=choice)
        sample_np = np.array(sample.to_numpy().flat)
        model.update(sample_np, model.winner(sample_np), i, STEPS)
        if i % 1000 == 0 and i > 0:
            print("Step {}/{}".format(i, STEPS))
            fig = model_plot(model.get_weights(), dataset)
            fig.savefig(TRAINING_STEPS.format(i))
        if i % 1000 == BATCHSIZE / 2:
            dataset = load_random_selection_of_data()
    return

    plt.figure(0)
    for i in range(300):
        plotting(ax, geopot, index=i)
        plt.savefig(IMAGE_PATH)
        sleep(1)
        return


if __name__ == "__main__":
    main()
