
include("../src/BindingInference.jl")
import .BindingInference as ba

# Import Packages
println("Importing packages")
using Distributed; #  (nprocs() == 1 ? addprocs(4) : nothing); println("Processors = ", procs())
using HDF5
using SpecialFunctions
using Distributions
using SparseArrays
using SharedArrays 
using DistributedArrays
using Random
using LinearAlgebra
import PyPlot as plt; plt.pygui(true)

# Set up environment
@everywhere ENV["PATHTOSAMPLECACHE"] = "/Users/jbryaniv/Desktop/Projects/_samplecache_"
@everywhere ENV["PATHTOPRECOMPUTED"] = "/Users/jbryaniv/Desktop/Projects/_precomputed_/"

# Save name function
function get_savename(file; kwargs...)

    # Check extension
    if lowercase(file)[end-2:end] != ".h5"
        file = file * ".h5"
    end

    # Set savename
    savename = file[1:end-3] * "_INFERENCE"

    # Add kwargs
    if length(kwargs) > 0
        kwargs = Dict([(
            string(key), value) for (key, value) in pairs(kwargs) if (value !== nothing) & (value != false)
        ])
        kwargs = sort(collect(pairs(kwargs)), by=x->x[1])
        suffix = [key*"="*string(value) for (key, value) in kwargs]
        savename = savename * "_" * join(suffix, "_")
    end

    # Return savename
    return savename
end


# Run script
function get_data(
        file; path="/Users/jbryaniv/Desktop/Data/Binding/", num_rois=nothing, kwargs...
    )

    # Load data
    if lowercase(file)[end-2:end] != ".h5"
        file = file * ".h5"
    end
    fid = h5open(path *file, "r")
    data = read(fid["data"])'
    parameters = Dict([(string(key),read(fid[key])) for key in keys(fid) if key != "data"])
    close(fid)

    # Downsample data
    if num_rois !== nothing
        ids = shuffle(1:size(data, 1))[1:minimum((size(data, 1), num_rois))]
        data = data[ids, :]
    end

    # Set parameters
    parameters = merge(
        parameters,
        Dict([(string(key),value) for (key, value) in pairs(kwargs)])
    )

    return data, parameters
end

#### RUN SCRIPT ####

# Get system arg
ID = 3
if length(ARGS) > 0
    ID = parse(Int, ARGS[1]) + 1
end

# Get files and run parameters
path = "/Users/jbryaniv/Desktop/Data/Binding/"
files = [file for file in readdir(path) if endswith(lowercase(file), ".h5")]
files = [file for file in files if !contains(lowercase(file), "infereence")]
files = [file for file in files if !contains(lowercase(file), "jinxuan")]

# Run file and params
file = files[ID]
params = (
    num_rois=10,
)
savename = get_savename(file; params...)
data, parameters = get_data(file; params...)
data = data[:, 1:10:end]
data, parameters = ba.simulate_data()
println("$(file) $(savename)")

# # Plot data
data, parameters = ba.simulate_data(
    k_photo = []
)
# fig, ax = plt.subplots(1, 5, sharey=true, sharex=true)
# plt.ion()
# plt.show()
# fig.suptitle("$(file[9:end-3])")
# for i in 1:size(ax, 1)
#     ax[i].plot(data[i, :])
#     ax[i].set_title("ROI $i")
#     ax[i].set_xlabel("Time")
# end
# ax[1].set_ylabel("Intensity")

# Run file
println("Analyzing data")
# variables = SimpleNamespace(merge(ba.PARAMETERS, parameters))
variables =  ba.analyze(
    data, parameters,
    num_iterations=100,
    saveas="outfiles/"*savename,
    plot=true,
)
ba.plot_variables(data, variables)

# Extract results
dt = variables.dt
partitions = variables.partitions
degenerate_ids = variables.degenerate_ids
mu_photo = variables.mu_photo
mu_back = variables.mu_back
sigma_photo = variables.sigma_photo
sigma_back = variables.sigma_back
k_photo = variables.k_photo
num_rois = variables.num_rois
num_frames = variables.num_frames
num_photo = variables.num_photo
num_macro = variables.num_macro
num_unique = variables.num_unique
background_times = variables.background_times
macrostates = variables.macrostates
num_bound = variables.num_bound


