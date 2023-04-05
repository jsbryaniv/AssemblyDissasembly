
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

# Select file
path = "/Users/jbryaniv/Desktop/Data/Binding/"
files = [file for file in os.listdir(path) if file.lower().endswith(".h5")]
files = [file for file in files if "binding" in file]
file = files[0]

# Load data
with h5py.File(path + file, "r") as f:
    data = f["data"][()]
num_rois, num_data = data.shape
parameters = dict([
    ("laser_power", ones(num_rois)),
])
# # Plot data
# fig, ax = plt.subplots(1, 1)
# plt.ion()
# plt.show()
# ax.set_title(file)
# ax.set_xlabel("Time")
# ax.set_ylabel("Intensity")
# for i in range(10):
#     ax.cla()
#     ax.plot(data[i, :])
#     ax.set_ylim(0, np.max(data))
#     plt.pause(1)

savename = file[:-4] + "_example2"
data, variables = ba.analyze(
        data, parameters,
        num_iterations=20,
        saveas="outfiles/"*savename,
        plot=plot,
    )

print("Done")

