
import os
import h5py
import parse
import numpy as np
import pandas as pd
import scipy.ndimage as nd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image


def plot_data(data, parameters=None, rois=None, title=None):

    # Extract parameters
    default_parameters = {
        'dt': 1,
    }
    if parameters is None:
        parameters = {}
    parameters = {**default_parameters, **parameters}
    dt = parameters['dt']

    # Set ROIs to plot
    if rois is None:
        rois = [0]
    elif isinstance(rois, (int, float, complex)):
        rois = [rois]

    # Set up figure
    fig, ax = plt.subplots(len(rois), 1, squeeze=False, sharex=True)

    # Calculate values
    times = dt * np.arange(data.shape[1])

    # Plot data
    for i, r in enumerate(rois):
        ax[i, 0].plot(times, data[r, :])
        # ax[i, 0].set_title(f'ROI {r}')
        ax[i, 0].set_ylabel('Counts')
    
    # # Set legend
    # handles, labels = ax[0, 0].get_legend_handles_labels()
    # by_label = dict(zip(labels, handles))
    # ax[0, 0].legend(by_label.values(), by_label.keys())

    # Set axes
    fig.suptitle(title)
    ax[-1, 0].set_xlabel('Time (ns)')
    plt.tight_layout()
    plt.pause(.1)

    return


def load_data_jinxuan():

    # Open data
    path = "../data/binding/Raw/Jinxuan/"
    file = "6400_noOSS_WT.xlsx"
    data = np.asarray(pd.read_excel(path + file))

    # Create parameters
    parameters = {}

    # Crop data
    data = data[:, 45:]
    ids = np.where(np.max(data, axis=1) > 1000)[0]  # Exclude ROIs with just background
    data = data[ids, :]
    ids = np.where(np.sum(data> 1000, axis=1) > 100)[0]  # Exclude ROIs with short traces
    data = data[ids, :]

    return data, parameters


def load_data_Herten(folder, path=None):

    # Set path
    if path is None:
        path = ""

    # Load files
    files = os.listdir(path + folder)
    files = [file for file in files if file.lower().endswith('peak.csv')]

    # Loop through files
    data = []
    laser_power = []
    surface_number = []
    for file in files:

        # Load data
        data_f = pd.read_csv(path + folder + "/" + file, header=1)
        data_f = np.asarray(data_f)
        data_f = data_f[:, 15:]

        # Get parameters
        num_rois_f = data_f.shape[0]
        laser_power_f = float(file.split('_')[2])
        surface_number_f = float(file.split('_')[0][7:])
        
        # Append data
        data.append(data_f)
        laser_power.append([laser_power_f]*num_rois_f)
        surface_number.append([surface_number_f]*num_rois_f)
    
    # Concatenate data
    data = np.vstack(data)
    laser_power = np.concatenate(laser_power)
    surface_number = np.concatenate(surface_number)

    # Get parameters from file name
    dt = float(files[0].split('_')[3])
    gain = float(files[0].split('_')[4])
    parameters = {
        'dt': dt,
        'gain': gain,
        'laser_power': laser_power,
        'surface_number': surface_number,
    }

    # Return data and parameters
    return data, parameters


def save_data(data, parameters, savename, path=None):

    # Set path
    if path is None:
        path = "../data/binding/"

    # Save savename
    if not savename.lower().endswith('.h5'):
        savename += '.h5'

    # Save data
    h5 = h5py.File(path + savename, 'w')
    h5.create_dataset('data', data=data)
    for key, value in parameters.items():
        h5.create_dataset(key, data=value)
    h5.close()

    return



if __name__ == "__main__":


    # Set paths
    path_data = "/Users/jbryaniv/Desktop/Data/Binding/"
    path_tif = f"{path_data}/ST051_220311_Dynamic_DNA_bleaching/"


    # Loop through Herten folders
    files = os.listdir(path_data)
    files = [f for f in files if f.endswith('mer.h5')]
    for file in files:
            
        # Load data
        h5 = h5py.File(path_data + file, 'r')
        data = h5["data"][()]
        parameters = {key: h5[key][()] for key in h5.keys() if key != "data"}
        h5.close()

        print(f"{file} :: {data.shape}")

        # Plot data
        for laser_power in np.unique(parameters['laser_power']):
            for surface_number in np.unique(parameters['surface_number']):
                savename = f"{file[:-3]}_W{laser_power}_S{surface_number}"
                ids = np.where(
                    (parameters['laser_power'] == laser_power) 
                    & (parameters['surface_number'] == surface_number)
                )[0]
                rois = np.random.choice(ids, min((5, len(ids))), replace=False)
                plot_data(data, parameters=parameters, rois=rois, title=savename)
                plt.savefig(f"{path_data}/{savename}.png")
                plt.close()