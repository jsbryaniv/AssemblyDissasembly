
import os
import h5py
import parse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture


### DEFINE FUNCTIONS ###

# Load binding data function
def open_StanData(filepath):

    # Parse file name
    """
    SurfaceXX_YYY_PPP_WW_GG_TT_peak.csv
    XX  = surface number
    YYY = laser wavelength
    PPP = power in percentage of max power
    GG  = EM gain
    WW  = ?
    TT  = Number of trace in the same region/surface.
    """
    filevars = parse.parse("surface{XX}_{YYY}_{PPP}_{GG}_{WWTT}_peak.csv", filepath.split("/")[-1])
    surface_number = int(filevars["XX"])
    laser_wavelength = int(filevars["YYY"])
    laser_power = int(filevars["PPP"])
    EM_gain = int(filevars["GG"])
    
    # Load data
    df = pd.read_csv(filepath, header=1)
    x_locs = df["x [nm]"].values
    y_locs = df["y [nm]"].values
    data = df.iloc[:, 10:].values

    # Load metadata
    with open(filepath) as f:
        metadata = f.readline()
    metadata = eval(metadata[1:])

    # Update metadata
    metadata["x_locs"] = x_locs
    metadata["y_locs"] = y_locs
    metadata["EM_gain"] = EM_gain
    metadata["laser_power"] = laser_power
    metadata["surface_number"] = surface_number
    metadata["laser_wavelength"] = laser_wavelength

    # fig, ax = plt.subplots(1, 1)
    # times = np.arange(data.shape[1])*.1
    # ax.set_ylabel("Intensity (ADU)")
    # ax.set_xlabel("Time (s)")
    # ax.plot(times, data[0, :], color='g')

    return data, metadata


# Get file paths
path_data = "/Users/jbryaniv/Desktop/Data/Binding/Raw/"
files = os.listdir(path_data)
files = [x for x in files if x.lower().startswith("stan")]
files = [f"{x}/{y}" for x in files for y in os.listdir(path_data + x)]
files = [x for x in files if os.path.isdir(path_data+x)]
files = [f"{x}/{y}" for x in files for y in os.listdir(path_data + x)]
files = [x for x in files if x.lower().endswith("_peak.csv")]

# Loop through files
for file in files:

    # Get data
    data, metadata = open_StanData(path_data+file)

    # Get concentration
    if "nM" in file:
        concentration = int(file.split("nM")[0].split("_")[-1])
    elif "blink" in file.lower():
        """
        surface 1-4 1nM
        surface 5-8 2nM
        surface 9-12 5nM
        surface 13-16 10nM
        """
        x = [0, 1, 1, 1, 1, 2, 2, 2, 2, 5, 5, 5, 5, 10, 10, 10, 10]
        concentration = x[int(file.split("surface")[1].split("_")[0])]
    elif "bleach" in file.lower():
        """
        Surface 1: 1nM
        Surface 2: 2nM
        Surface 3: 5nM
        Surface 4: 10nM
        """
        x = [0, 1, 2, 5, 10]
        concentration = x[int(file.split("surface")[1].split("_")[0])]
    metadata["concentration"] = concentration

    # Save data
    path_save = "/Users/jbryaniv/Desktop/Data/Binding/"
    savename = f"data_{file.replace('/', '_')}".replace(".csv", ".h5")
    with h5py.File(path_save + savename, "w") as f:
        f.create_dataset("data", data=data)
        for key, value in metadata.items():
            f.create_dataset(key, data=value)

    # Completed
    print(f"Done: {data.shape}") # {savename}")



print("Done!")

