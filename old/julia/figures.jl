
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

# Set up environment
println("Setting up environment")
cd(@__DIR__)
Pkg.activate(".")  # Activate the package environment before adding processors
if nprocs() == 1
    addprocs(5)
    println("Processors = $(procs())")
end
import BindingInference as ba
@everywhere workers() using BindingInference
@everywhere ENV["PATHTOSAMPLECACHE"] = "Outfiles/"


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

# Get files
path = "../../Data/Binding/"
files = [file for file in readdir(path) if endswith(lowercase(file), ".h5")]
NUM_ROIS = [10, 100, 200]
params_list = [(num_rois=num_rois,) for num_rois in NUM_ROIS]
file, params = [(file, params) for params in params_list for file in files][ID]
savename = get_savename(file; params...)
println("$(file) $(savename)")

# Get data
data, parameters = get_data(file; params...)
if "amplitude" in keys(parameters)
    parameters["laser_power"] = parameters["amplitude"]/mean(parameters["amplitude"])
end

samplecache = SampleCache(savename)
variables = SimpleNamespace(samplecache.get("MAP"))

mu_micro_samples = samplecache.get("mu_micro")
mu_back_samples = samplecache.get("mu_back")
k_micro_samples = samplecache.get("k_micro")
k_bind_samples = samplecache.get("k_bind")
pi_micro_samples = copy(k_micro_samples)
KD_samples = k_bind_samples[:, 1, 2] ./ k_bind_samples[:, 2, 1]
num_iterations = size(k_micro_samples, 1)
for iteration in 1:num_iterations
    pi_micro_samples[iteration, 1:end-1, :] = exp(pi_micro_samples[iteration, 1:end-1, :])
end

sc1 = SampleCache("binding_1nM_INFERENCE_num_rois=10")
sc5 = SampleCache("binding_5nM_INFERENCE_num_rois=10")
sc10 = SampleCache("binding_10nM_INFERENCE_num_rois=10")
variables1 = SimpleNamespace(sc1.get("MAP"))
variables5 = SimpleNamespace(sc5.get("MAP"))
variables10 = SimpleNamespace(sc10.get("MAP"))


# julia> variables1.k_bind
# 3×2 Matrix{Float64}:
#  -0.00919584   0.00919584
#   0.00479311  -0.00479311
#   0.1          0.9

# julia> variables5.k_bind
# 3×2 Matrix{Float64}:
#  -0.00844282   0.00844282
#   0.00477092  -0.00477092
#   0.1          0.9

# julia> variables10.k_bind
# 3×2 Matrix{Float64}:
#  -0.00479081   0.00479081
#   0.00419573  -0.00419573
#   0.1          0.9


# Bind rates
#   1nM   5nM  10nM
# .0048 .0048 .0042
# Unbind rates
#   1nM   5nM  10nM
# .0092 .0084 .0048

