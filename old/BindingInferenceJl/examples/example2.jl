
# Import packages and set up environment
println("Importing packages")
using Pkg
using Distributed
using HDF5
using SpecialFunctions
using Distributions
using SparseArrays
using SharedArrays 
using DistributedArrays
using Random
using LinearAlgebra
using SampleCaches
using SimpleNamespaces
import PyPlot as plt; plt.pygui(true)

# Testing

# Set up environment
println("Setting up environment")
runlocal = true
if runlocal
    Pkg.activate(".")
else
    Pkg.add(url="https://github.com/LabPresse/BindingInference.git")
end
if nprocs() == 1
    addprocs(5); println("Processors = $(procs())")
end
import BindingInference as ba
@everywhere workers() using BindingInference
@everywhere ENV["PATHTOSAMPLECACHE"] = "/Users/jbryaniv/Desktop/Projects/_samplecache_"
@everywhere ENV["PATHTOPRECOMPUTED"] = "/Users/jbryaniv/Desktop/Projects/_precomputed_/"
@everywhere ENV["PATHTOSAMPLECACHE"] = "outfiles/"


### HELPER FUNCTIONS ###

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

# Get data function
function get_data(
        file; path="/Users/jbryaniv/Desktop/Data/Binding/", 
        num_rois=nothing, downsample=nothing, kwargs...
    )

    # Load data
    if lowercase(file)[end-2:end] != ".h5"
        file = file * ".h5"
    end
    fid = h5open(path *file, "r")
    data = read(fid["data"])'
    parameters = Dict{String, Any}([(string(key),read(fid[key])) for key in keys(fid) if key != "data"])
    close(fid)

    # Downsample data
    if downsample !== nothing
        data = data[:, 1:downsample:end]
        parameters["downsample"] = downsample
    end
    if num_rois !== nothing
        ids = shuffle(1:size(data, 1))[1:minimum((size(data, 1), num_rois))]
        data = data[ids, :]
        for key in keys(parameters)
            try
                parameters[key] = parameters[key][ids]
            catch
            end
        end
    end

    # Set parameters
    parameters = merge(
        parameters,
        Dict([(string(key),value) for (key, value) in pairs(kwargs)])
    )

    return data, parameters
end

# Plot data function
function plot_data(data; rois=1)

    # If rois is a number encase in a list
    if typeof(rois) <: Number
        rois = [rois]
    end

    # Create figure
    fig, ax = plt.subplots(1, len(rois), sharey=true, sharex=true)
    plt.ion()
    plt.show()
    
    # Plot data
    for (r, i) in enumerate(rois)
        ax[i].plot(data[i, :])
        ax[i].set_title("ROI $r")
        ax[i].set_xlabel("Time")
    end
    ax[1].set_ylabel("Intensity")

    return nothing
end

#### RUN SCRIPT ####

# Get system arg
ID = 4
if length(ARGS) > 0
    ID = parse(Int, ARGS[1]) + 1
end

# Get files and run parameters
path = "/Users/jbryaniv/Desktop/Data/Binding/"
files = [file for file in readdir(path) if endswith(lowercase(file), ".h5")]
files = [file for file in files if !contains(file, "INFERENCE")]
files = [file for file in files if !contains(lowercase(file), "jinxuan")]
file = files[ID]
params = []  # (num_rois=100,)
savename = get_savename(file; params...)

# Run file and params
println("$(file) $(savename)")
data, parameters = get_data(file; params...)
if "amplitude" in keys(parameters)
    parameters["laser_power"] = parameters["amplitude"]/mean(parameters["amplitude"])
end

# Run file
println("Analyzing data")
variables = ba.analyze(
    data, parameters,
    num_iterations=1000,
    saveas=savename,
    plot=false,
)
if runlocal
    ba.plot_variables(data, variables)
end


samplecache = SampleCache(saveas)
k_micro_samples = samplecache.get("k_micro")
k_bind_samples = samplecache.get("k_bind")
pi_micro_samples = copy(k_micro_samples)
KD_samples = k_bind_samples[:, 1, 2] ./ k_bind_samples[:, 2, 1]
num_iterations = size(k_micro_samples, 1)
for iteration in 1:num_iterations
    pi_micro_samples[iteration, 1:end-1, :] = exp(pi_micro_samples[iteration, 1:end-1, :])
end


# # Extract results
# dt = variables.dt
# partitions = variables.partitions
# degenerate_ids = variables.degenerate_ids
# mu_micro = variables.mu_micro
# mu_back = variables.mu_back
# sigma_micro = variables.sigma_micro
# sigma_back = variables.sigma_back
# k_micro = variables.k_micro
# k_bind = variables.k_bind
# concentration = variables.concentration
# laser_power = variables.laser_power
# num_rois = variables.num_rois
# num_frames = variables.num_frames
# num_micro = variables.num_micro
# num_macro = variables.num_macro
# num_unique = variables.num_unique
# macrostates = variables.macrostates
# num_bound = variables.num_bound

# r = 1
# pi_macro = micro_rates_to_macro_transitions(
#     dt, k_micro, k_bind, C=concentration[r], W=laser_power[r], partitions=partitions
# )

