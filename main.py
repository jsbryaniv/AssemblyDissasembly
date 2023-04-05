
# Import standard modules
import os
import sys
import h5py
import copy
import time
import numpy as np
import numba as nb
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from scipy import stats
from types import SimpleNamespace
from BindingInference import BindingInference


# Load data function
def load_data(file):

    # Check file name
    if not file.lower().endswith(".h5"):
        file += ".h5"
    
    # Load data
    h5 = h5py.File(file, "r")
    data = h5["data"][()]
    metadata = {key: h5[key][()] for key in h5.keys() if key != "data"}
    h5.close()

    return data, metadata

# Main script
if __name__ == "__main__":

    # Get files
    path = "../../Data/Binding/"
    files = [file for file in os.listdir(path) if file.lower().endswith(".h5")]

    # Loop through files
    for i, file in enumerate(files):
        print(f"Processing {i}/{len(files)} : {file}...")

        # Load data
        data, parameters = load_data(path + file)
        num_rois, num_frames = data.shape

        # # Loop through ROIs
        # for roi in range(num_rois):
        #     data_r = data[roi, :]
        #     variables = BindingInference.analyze_data(data_r, parameters)
        #     states[roi, :] = variables.s
        
        # Define sampling function
        def sample_states(args):
            r, data_r, parameters = args
            states = BindingInference.analyze_data(data_r, parameters).s
            if r % 10 == 0:
                print(f"- {r}/{num_rois}")
            return states
        
        # Sample states in parallel
        params = [(r, data[r, :], parameters) for r in range(num_rois)]
        states = Parallel(n_jobs=-1)(delayed(sample_states)(x) for x in params)
        states = np.vstack(states)

        # Save data
        savename = file.replace(".h5", "_states.h5")
        h5 = h5py.File("Outfiles/" + savename, "w")
        h5.create_dataset("states", data=states)
        for key, value in parameters.items():
            h5.create_dataset(key, data=value)
        h5.close()

    print("Done!")

