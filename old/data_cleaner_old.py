
import os
import h5py
import parse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture


### DEFINE FUNCTIONS ###

# Load binding data function
def open_StanData(file, path=None):

    # Get path
    if path is None:
        path = ""

    # Load data
    df = pd.read_csv(path+file, header=1)
    x_locs = df["x [nm]"].values
    y_locs = df["y [nm]"].values
    data = df.iloc[:, 10:].values

    # Load metadata
    with open(path+file) as f:
        metadata = f.readline()
    metadata = eval(metadata[1:])

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
    surface_number = int(filevars["XX"]) * np.ones(data.shape[0])
    laser_wavelength = int(filevars["YYY"]) * np.ones(data.shape[0])
    laser_power = int(filevars["PPP"]) * np.ones(data.shape[0])
    EM_gain = int(filevars["GG"]) * np.ones(data.shape[0])

    # Update metadata
    metadata["x_locs"] = x_locs
    metadata["y_locs"] = y_locs
    metadata["EM_gain"] = EM_gain
    metadata["laser_power"] = laser_power
    metadata["surface_number"] = surface_number
    metadata["laser_wavelength"] = laser_wavelength
    for key, value in metadata.items():
        metadata[key] = value * np.ones(data.shape[0])

    fig, ax = plt.subplots(1, 1)
    times = np.arange(data.shape[1])*.1
    ax.set_ylabel("Intensity (ADU)")
    ax.set_xlabel("Time (s)")
    ax.plot(times, data[0, :], color='g')

    return data, metadata

# Filter data function
def filter_data(data, metadata, spikes=True, twostate=True, positions=True, jumps=True):

    # Get metadata
    x_locs = metadata["x_locs"]
    y_locs = metadata["y_locs"]
    pix_size = metadata["pix_size"]
    EM_gain = metadata["EM_gain"]
    laser_power = metadata["laser_power"]
    trace_number = metadata["trace_number"]
    surface_number = metadata["surface_number"]
    laser_wavelength = metadata["laser_wavelength"]
    concentration = metadata["concentration"]

    # Create keep array
    keep = np.ones(data.shape[0], dtype=bool)

    # Filter out spikes
    if spikes:
        ids = (
            ~((np.max(data, axis=1) - np.mean(data, axis=1)) > 4*np.std(data, axis=1))
            &
            ~((np.mean(data, axis=1) - np.min(data, axis=1)) > 4*np.std(data, axis=1))
        )
        keep[ids] = False

    # Filter out x and y locations
    if positions is not None:
        xlims = 512*np.array([.33, .66])
        ylims = 512*np.array([.33, .66])
        ids = (
            (x_locs/pix_size > xlims[0]) & (x_locs/pix_size < xlims[1])
            &
            (y_locs/pix_size > ylims[0]) & (y_locs/pix_size < ylims[1])
        )
        keep[ids] = False

    # Filter out large jumps
    if jumps:
        jumps = data[:, 2:] - data[:, 1:-1]
        ids = np.argsort(np.max(np.abs(jumps), axis=1))[:int(.9*jumps.shape[0])]
        keep[ids] = False

    # Get amplitude
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
        gmm = GaussianMixture(n_components=n_components)
        gmm.fit(data_r)
        means = np.sort(gmm.means_[:, 0])
        amplitude[r] = means[-1] - means[0]

        # Filter out rows with many components
        if twostate:
            if (n_components > 2) or (amplitude[r] == 0):
                keep[r] = False
    metadata["amplitude"] = amplitude

    # Filter out rows
    data = data[keep, :]
    for key, value in metadata.items():
        try:
            metadata[key] = value[keep]
        except:
            pass

    return data, metadata


# ### CONVERT TO HDF5 ###

# Loop through concentrations
datasets = [
    # "StanData2/results_max_1nM",
    # "StanData2/results_max_5nM",
    # "StanData2/results_max_10nM",
    "StanDataBlinking/results_max",
    # "StanDataBleaching/LT1A2_A3_B2_B3_bleaching_with_dye_50pct",
    # "StanDataBleaching/LT1A2_A3_B2_B3_bleaching_with_dye_100pct",
]
for dataset in datasets:

    # Get files from dataset
    datapath = "/Users/jbryaniv/Desktop/Data/Binding/Raw/"
    files = [file for file in os.listdir(datapath+dataset) if file.lower().endswith("_peak.csv")]

    # Loop through files
    for file in files:

        # Get data
        data, metadata = open_StanData(file, path=datapath + dataset + '/')

        # Get concentration
        if dataset.split("/")[0] == "StanData2":
            nM = int(parse.parse("StanData2/results_max_{nM}nM", dataset)["nM"])
            concentration = nM * np.ones(data.shape[0])
        else:
            concentration = np.ones(data.shape[0])
            for (s, nM) in enumerate([0, 1, 2, 5, 10]):
                ids = np.where(metadata["surface_number"] == s)[0]
                concentration[ids] = nM
        metadata["concentration"] = concentration

        # Save data
        savename = datapath + f"dataset_{file}.h5"
        with h5py.File(savename, "w") as f:
            f.create_dataset("data", data=data)
            for key, value in metadata.items():
                f.create_dataset(key, data=value)

    # # Plot data
    # fig = plt.gcf()
    # fig.clf()
    # ax = fig.add_subplot(111)
    # ax.set_xlabel("Time")
    # ax.set_ylabel("Intensity")
    # for i in range(100):  # data.shape[0]):
    #     ax.cla()
    #     # Get state levels
    #     data_r = data[i, :].reshape(-1, 1)
    #     bic = []
    #     for n_components in range(1, 4):
    #         gmm = GaussianMixture(n_components=n_components)
    #         gmm.fit(data_r)
    #         bic.append(gmm.bic(data_r))
    #     n_components = np.argmin(bic) + 1
    #     gmm = GaussianMixture(n_components=n_components)
    #     gmm.fit(data_r)
    #     # Plot
    #     ax.set_ylim([1000, 10000])
    #     ax.plot(data[i, :])
    #     ax.hlines(gmm.means_, 1000, 10000, colors="r")
    #     plt.pause(10/data.shape[0])
    # plt.pause(1)

    # Completed
    print(f"Done: ") # {savename}")



### CONSOLIDATE HDF5 FILES ###

datasets = [
    "StanData2",
    "StanDataBlinking",
    "StanDataBleaching",
]
for dataset in datasets:

    # Set filters
    filters = {"spikes":True, "twostate":True, "positions": False, "jumps": False}

    # Get files
    path = "/Users/jbryaniv/Desktop/Data/Binding/Raw/"
    files = os.listdir(path + dataset)
    files = [f for f in files if f.endswith(".h5")]

    # Load data from first file
    h5 = h5py.File(f"{path}{dataset}/{files[0]}", "r")
    data = h5["data"][:]
    metadata = {key: h5[key][()] for key in h5.keys() if key != "data"}
    h5.close()
    data, metadata = filter_data(data, metadata, **filters)

    # Load data from remaining files
    for f in files[1:]:
        h5 = h5py.File(f"{path}{dataset}/{f}", "r")
        data_r = h5["data"][:]
        metadata_r = {key: h5[key][()] for key in h5.keys() if key != "data"}
        h5.close()
        data_r, metadata_r = filter_data(data_r, metadata_r, **filters)
        data = np.vstack((data, data_r))
        for key in metadata_r.keys():
            if key != "data":
                metadata[key] = np.hstack((metadata[key], metadata_r[key]))
        h5.close()

    # Save data
    savepath = "/Users/jbryaniv/Desktop/Data/Binding/"
    savename = "_".join([dataset, *[f"{key}={val}" for key, val in filters.items() if val]]) + ".h5"
    h5 = h5py.File(f"{savepath}{dataset}.h5", "w")
    h5.create_dataset("data", data=data)
    for key, value in metadata.items():
        h5.create_dataset(key, data=value)
    h5.close()

    # Completed
    print(f"Done: {dataset}")


print("done")

