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

def plotting(ax, data):
    width = len(data[0])
    height = len(data)
    x_range = (COORDS[1] - COORDS[0])
    y_range =  (COORDS[3] - COORDS[2])
    x = [(i + 0.5) / width * x_range + COORDS[0] for i in range(width)]
    y = [(i + 0.5) / width * y_range + COORDS[2] for i in range(height)]
    print(x)
    print(y)
    ax.pcolor(x, y, data)


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

    data = xr.load_dataset("geopot_93-22.grib")
    plot = data.t2m[0].plot(
        cmap=plt.cm.coolwarm, transform=ccrs.PlateCarree()
    )

    plotting(ax, harvest)

    plt.show()
    plt.savefig("/mnt/c/Users/Julian/Desktop/test.png")


if __name__ == "__main__":
    main()
