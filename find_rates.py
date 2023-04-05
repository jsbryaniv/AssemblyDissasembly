
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
from BindingInference import StateInference, RateInference


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

# Load results function
def load_results(file):
    
    # Check file name
    if not file.lower().endswith(".h5"):
        file += ".h5"
    
    # Load data
    h5 = h5py.File(file, "r")
    states = h5["states"][()]
    metadata = {key: h5[key][()] for key in h5.keys() if key != "states"}
    h5.close()

    return states, metadata

# Main script
if __name__ == "__main__":

    # Print status
    print("Running find_rates.py...")

    # Set up paths
    path_data = "../../Data/Binding/"
    path_results = "Outfiles/"
    path_figures = "Figures/"

    # Get files
    files = [file for file in os.listdir(path_results) if file.lower().endswith(".h5")]
    experiments = [file[5:] for file in files]
    experiments = ["_".join(file.split("_")[:2]) for file in experiments]
    experiments = list(set(experiments))

    # Loop through experiments
    for exp, experiment in enumerate(experiments):

        # Get files
        files_exp = [file for file in files if experiment in file]

        # Set up data
        data = None
        parameters = {}

        # Loop through files
        for i, file in enumerate(files_exp):
            print(f"Processing {i}/{len(files_exp)} : {file}...")

            # Load results and data
            states_i, _ = load_results(path_results + file)
            data_i, parameters_i = load_data(path_data + file.replace("_states", ""))
            l = parameters_i["laser_power"]
            c = parameters_i["concentration"]

            # Filter ROIs
            ids = np.where(np.max(states_i, axis=1) < 5)[0]
            states_i = states_i[ids, :]
            num_rois = states_i.shape[0]

            # Add to data
            if data is None:
                data = states_i
                parameters["dt"] = 1
                parameters["laserpowers"] = l * np.ones(num_rois)
                parameters["concentrations"] = c * np.ones(num_rois)
            else:
                data = np.vstack((data, states_i))
                parameters["laserpowers"] = np.hstack(
                    (parameters["laserpowers"], l * np.ones(num_rois))
                )
                parameters["concentrations"] = np.hstack(
                    (parameters["concentrations"], c * np.ones(num_rois))
                )

            # Analyze one example trace per file
            print(f"Analyzing {file}...")
            variables_i = StateInference.analyze_data(data_i[0, :], parameters_i)
            StateInference.plot_data(data_i[0, :], variables_i)
            plt.suptitle(experiment + " : " + file)
            plt.gcf().set_size_inches(10, 3)
            plt.tight_layout()
            plt.pause(.1)
            plt.savefig(path_figures + f"{experiment}_{file}_trace.png")

        # Get rates
        print(f"Analyzing {experiment}...")
        variables, Samples = RateInference.analyze_data(data, parameters)

        # Save results
        print(f"Saving {experiment}...")
        RateInference.plot_rates(variables, Samples)
        plt.suptitle(experiment)
        plt.gcf().set_size_inches(10, 5)
        plt.tight_layout()
        plt.pause(.1)
        plt.savefig(path_figures + f"{experiment}_rates.png")

        # Done
        print(f"{exp}/{len(experiments)} : {file} done!")

    print("Done!")

# variables.k_on
# 4.532774736986114e-06
# variables.k_off
# 0.002384444405601617
# variables.k_photo
# 5.091206925388496e-08
