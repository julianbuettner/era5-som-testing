import somoclu
from time import sleep
from matplotlib.offsetbox import AnchoredText
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature

COORDS = [-20, 50, 30, 80]

harvest = np.array([[0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
                    [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0],
                    [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0],
                    [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0],
                    [0.7, 1.7, 0.6, 2.6, 2.2, 6.2, 0.0],
                    [1.3, 1.2, 0.0, 0.0, 0.0, 3.2, 5.1],
                    [0.1, 2.0, 0.0, 1.4, 0.0, 1.9, 6.3]])
harvest = np.random.rand(100, 100)
harvest = np.random.rand(COORDS[1] - COORDS[0], COORDS[3] - COORDS[2])

IMAGE_PATH = "/mnt/c/Users/Julian/Desktop/test.png"

def plotting(ax, da: xr.DataArray, index=0):
    # width = len(data[0])
    # height = len(data)
    # x_range = (COORDS[1] - COORDS[0])
    # y_range =  (COORDS[3] - COORDS[2])
    # print(x)
    # print(y)
    # print(da.values[0])
    x = da.coords["latitude"].values
    y = da.coords["longitude"].values
    print("Dim:", len(x) * len(y))
    # print(x, y, da.values[index].shape)
    # X, Y = np.meshgrid(da.coords["latitude"].values, da.coords["longitude"].values)
    ax.pcolormesh(
        [yy for yy in y],
        [xx for xx in x],
        # da.values[0],
        da.values[index][:-1, :-1],
        shading='nearest',
    )

def model_plot(model):
    ...


def resample(a: xr.Dataset):
    a = a.resample(time="1D").mean()
    a = a.coarsen(longitude=4, boundary='trim').mean()
    a = a.coarsen(latitude=4, boundary='trim').mean()
    return a


def main():
    # plt.style.use('_mpl-gallery-nogrid')
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent(COORDS, crs=ccrs.PlateCarree())

    # Put a background image on for nice sea rendering.
    ax.stock_img()

    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.COASTLINE)
    ax.coastlines(resolution="50m")

    SOURCE = "Natural Earth"
    LICENSE = "public domain"
    # Add a text annotation for the license information to the
    # the bottom right corner.
    text = AnchoredText(
        "\u00A9 {}; license: {}" "".format(SOURCE, LICENSE),
        loc=4,
        prop={"size": 12},
        frameon=True,
    )
    # ax.add_artist(text)

    data = xr.load_dataset("test500.nc")
    geopot = data["z"]
    print(geopot.to_numpy().shape)
    print("Normalize...")
    geopot = resample(geopot)
    data_dim = geopot.to_numpy().shape
    print("Datapoint dim", data_dim)

    model = somoclu.Somoclu(4, 4)
    sample = geopot.to_numpy().reshape((data_dim[0], data_dim[1] * data_dim[2],))
    print("Sample", sample.shape)
    model.train(sample, epochs=10000)
    # print("BMUs:", model.get)
    model.view_umatrix(bestmatches=True, filename=IMAGE_PATH)

    # for i in range(300):
    #     plotting(ax, geopot, index=i)
    #     plt.savefig("out.png")
    #     sleep(1)
    # plt.savefig("/mnt/c/Users/Julian/Desktop/test.png")


if __name__ == "__main__":
    main()
