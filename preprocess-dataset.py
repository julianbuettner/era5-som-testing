
import numpy as np
import xarray as xr
from xarray.core.dataarray import DataArray
from itertools import product
from random import randint

IMAGE_PATH = "/mnt/c/Users/Julian/Desktop/test.png"
IMAGE_PATH = "test.png"
COORDS = None

DATASET = "full500.nc"
# DATASET = "data500.nc"
SAMPLES_PER_DAY = 4

def resample(a: xr.Dataset):
    a = a.resample(time="1D", skipna=True).mean()
    a = a.coarsen(longitude=2 * 2, boundary="trim").mean()
    a = a.coarsen(latitude=2 * 2, boundary="trim").mean()
    return a

def load_dataset():
    data = xr.open_dataset(DATASET)
    print("Data points:", len(data["time"]))
    x = len(data["time"]) / SAMPLES_PER_DAY
    print(f"With {SAMPLES_PER_DAY} samples per day: {x} days")
    return resample(data)


def main():
    global COORDS
    dataset = load_dataset()
    print(dataset)

    dataset_np = dataset["z"].as_numpy()
    count_ok = 0
    count_nan_inf = 0
    print("Counting ok layers...")
    for layer in dataset_np:
        if np.isnan(layer).any():
            count_nan_inf += 1
            continue
        if not np.isfinite(layer).all():
            count_nan_inf += 1
            continue
        count_ok += 1
    print()
    print(f"Ok: {count_ok}, skipped: {count_nan_inf}")

    dataset.to_netcdf("resampled.nc")

if __name__ == "__main__":
    main()
