
import os
import h5py
import parse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# Load binding data function
def open_StanData_old(nM):

    # Select file
    path = f"/Users/jbryaniv/Desktop/Data/Binding/Raw/StanData2/results_max_{nM}nM/"
    file = [file for file in os.listdir(path) if file.lower().endswith("_peak.csv")][0]

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
    filevars = parse.parse("surface{XX}_{YYY}_{PPP}_{GG}_{WWTT}_peak.csv", file)
    surface_number = int(filevars["XX"])
    laser_wavelength = int(filevars["YYY"])
    laser_power = int(filevars["PPP"])
    EM_gain = int(filevars["GG"])
    trace_number = filevars["WWTT"]

    # Load metadata
    with open(path+file) as f:
        metadata = f.readline()
    metadata = eval(metadata[1:])

    # Load data
    df = pd.read_csv(path + file, header=1)
    x_locs = df["x [nm]"].values
    y_locs = df["y [nm]"].values
    data = df.iloc[:, 10:].values

    # Filter out spikes
    ids = (
        ~((np.max(data, axis=1) - np.mean(data, axis=1)) > 4*np.std(data, axis=1))
        &
        ~((np.mean(data, axis=1) - np.min(data, axis=1)) > 4*np.std(data, axis=1))
    )
    data = data[ids, :]
    x_locs = x_locs[ids]
    y_locs = y_locs[ids]

    # Filter out x and y locations
    xlims = np.array([.33, .66])*512*metadata["pix_size"]
    ylims = np.array([.33, .66])*512*metadata["pix_size"]
    ids = (
        (x_locs > xlims[0]) & (x_locs < xlims[1])
        &
        (y_locs > ylims[0]) & (y_locs < ylims[1])
    )
    data = data[ids, :]
    x_locs = x_locs[ids]
    y_locs = y_locs[ids]

    # Filter out large jumps
    jumps = data[:, 2:] - data[:, 1:-1]
    ids = np.argsort(np.max(np.abs(jumps), axis=1))[:int(.9*jumps.shape[0])]
    data = data[ids, :]
    x_locs = x_locs[ids]
    y_locs = y_locs[ids]

    # Update metadata
    metadata["x_locs"] = x_locs
    metadata["y_locs"] = y_locs
    metadata["EM_gain"] = EM_gain
    metadata["laser_power"] = laser_power
    metadata["trace_number"] = trace_number
    metadata["surface_number"] = surface_number
    metadata["laser_wavelength"] = laser_wavelength
    metadata["concentration"] = nM

    return data, metadata

def open_StanData(nM):

    # Select file
    path = f"/Users/jbryaniv/Desktop/Data/Binding/Raw/StanData2/results_max_{nM}nM/"
    file = [file for file in os.listdir(path) if file.lower().endswith("_peak.csv")][0]

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
    filevars = parse.parse("surface{XX}_{YYY}_{PPP}_{GG}_{WWTT}_peak.csv", file)
    surface_number = int(filevars["XX"])
    laser_wavelength = int(filevars["YYY"])
    laser_power = int(filevars["PPP"])
    EM_gain = int(filevars["GG"])
    trace_number = filevars["WWTT"]

    # Load metadata
    with open(path+file) as f:
        metadata = f.readline()
    metadata = eval(metadata[1:])

    # Load data
    df = pd.read_csv(path + file, header=1)
    x_locs = df["x [nm]"].values
    y_locs = df["y [nm]"].values
    data = df.iloc[:, 10:].values

    # Loop through each row
    keep = np.ones(data.shape[0], dtype=bool)
    amplitude = np.zeros(data.shape[0])
    for r in range(data.shape[0]):

        # Select row
        data_r = data[r, :].reshape(-1, 1)

        # Find the optimal number of components
        bic = []
        for n_components in range(1, 4):
            gmm = GaussianMixture(n_components=n_components)
            gmm.fit(data_r)
            bic.append(gmm.bic(data_r))
        n_components = np.argmin(bic) + 1

        # Get amplitude 
        gmm = GaussianMixture(n_components=2)
        gmm.fit(data_r)
        means = np.sort(gmm.means_[:, 0])
        amplitude[r] = means[-1] - means[0]

        # Filter out rows with many components
        if n_components > 2:
            keep[r] = False
        if amplitude[r] == 0:
            keep[r] = False

    # Filter out rows
    data = data[keep, :]
    x_locs = x_locs[keep]
    y_locs = y_locs[keep]
    amplitude = amplitude[keep]

    # Update metadata
    metadata["x_locs"] = x_locs
    metadata["y_locs"] = y_locs
    metadata["amplitude"] = amplitude
    metadata["EM_gain"] = EM_gain
    metadata["laser_power"] = laser_power
    metadata["trace_number"] = trace_number
    metadata["surface_number"] = surface_number
    metadata["laser_wavelength"] = laser_wavelength
    metadata["concentration"] = nM

    return data, metadata

# Loop through concentrations
for nM in [1, 5, 10]:

    # Load data
    data, metadata = open_StanData(nM)

    # Save data
    path = "/Users/jbryaniv/Desktop/Data/Binding/"
    savename = f"binding_{nM}nM.h5"
    with h5py.File(savename, "w") as f:
        f.create_dataset("data", data=data)
        for key, value in metadata.items():
            f.create_dataset(key, data=value)

    # Plot data
    fig = plt.gcf()
    fig.clf()
    ax = fig.add_subplot(111)
    ax.set_title(f"{nM}nM")
    ax.set_xlabel("Time")
    ax.set_ylabel("Intensity")
    for i in range(data.shape[0]):
        ax.plot(data[i, :])
        plt.pause(10/data.shape[0])
        ax.clear()
    plt.pause(1)

    # Completed
    print(f"Done: {savename}")



# Merge data into one file
data = []
parameters = {
    "x_locs": [],
    "y_locs": [],
    "amplitude": [],
    "EM_gain": [],
    "laser_power": [],
    "surface_number": [],
    "laser_wavelength": [],
    "concentration": [],
}
for nM in [1, 5, 10]:

    # Load data
    path = "/Users/jbryaniv/Desktop/Data/Binding/"
    savename = f"binding_{nM}nM.h5"
    h5 = h5py.File(savename, "r")
    data_r = h5["data"][()]
    num_rows = data_r.shape[0]
    parameters_r = {key: h5[key][()] for key in h5.keys() if key != 'data'}
    h5.close()

    # Append data
    data.append(data_r)
    for key, val in parameters_r.items():
        if key in parameters.keys():
            parameters[key].extend(np.ones(num_rows)*val)

# Save data
data = np.vstack(data)
path = "/Users/jbryaniv/Desktop/Data/Binding/"
savename = f"binding_AllnM.h5"
h5 = h5py.File(path+savename, "w")
h5.create_dataset("data", data=data)
for key, value in parameters.items():
    h5.create_dataset(key, data=value)
h5.close()

print("done")

