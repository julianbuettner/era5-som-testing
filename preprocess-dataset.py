
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
IMAGE_PATH = "test.png"
COORDS = None

DATASET = "full500.nc"
DATASET = "data500.nc"
TRAINING_STEPS = "ani1/training_{:05}.png"
FIG_FAC = 2
FIGSIZE = [6.4 * FIG_FAC, 4.8 * FIG_FAC]
STEPS = 90 * 1000
BATCHSIZE = 110

def resample(a: xr.Dataset):
    a = a.resample(time="1D").mean()
    a = a.coarsen(longitude=4 * 4, boundary="trim").mean()
    a = a.coarsen(latitude=4 * 4, boundary="trim").mean()
    return a

def load_dataset():
    data = xr.open_dataset(DATASET)
    # time_length = len(data["time"])
    # time_start = randint(0, time_length - BATCHSIZE - 1)
    # time_slice = slice(data["time"][time_start], data["time"][time_start + BATCHSIZE])
    # print("Time slice starting from", data["time"][time_start].as_numpy())
    # dataset = data.sel(time=time_slice)
    return resample(data)


def main():
    global COORDS
    dataset = load_dataset()
    dataset.to_netcdf("resampled.nc")

if __name__ == "__main__":
    main()
