
# Import packages and set up environment
println("Importing packages")
cd(@__DIR__)
using Pkg
using Revise
using Distributed
using HDF5
using SpecialFunctions
using Distributions
using SparseArrays
using SharedArrays 
using DistributedArrays
using Random
using LinearAlgebra
import PyPlot as plt; plt.pygui(true)

# Import BindingInference
println("Importing BindingInference")
Pkg.develop(path="./BindingInferenceJl/")
using BindingInference
StateFinder = BindingInference.StateFinder

# Add local packages
println("Adding local packages")
Pkg.develop(path="../Tools/SimpleNamespaces/")
Pkg.develop(path="../Tools/SampleCaches/")
Pkg.instantiate()
Pkg.resolve()
using SampleCaches
using SimpleNamespaces


# Set up environment
println("Setting up environment")
if nprocs() == 1
    addprocs(4, exeflags="--project=$(Base.active_project())")
    println("Processors = $(procs())")
end
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
        get_groundtruth=false, num_rois=nothing, downsample=nothing, kwargs...
    )

    # Load data
    if lowercase(file)[end-2:end] != ".h5"
        file = file * ".h5"
    end
    fid = h5open(path *file, "r")
    data = read(fid["data"])'
    parameters = Dict{String, Any}([(string(key),read(fid[key])) for key in keys(fid) if key != "data"])
    if get_groundtruth
        gt = fid["groundtruth"]
        groundtruth = Dict{String, Any}([(string(key),read(gt[key])) for key in keys(gt)])
    end
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

    # Return
    if get_groundtruth
        return data, parameters, groundtruth
    else
        return data, parameters
    end
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


# Get file
ID = 100
if length(ARGS) > 0
    # Get system arg if specified
    ID = parse(Int, ARGS[1]) + 1
end
path = "../../Data/Binding/"
files = [file for file in readdir(path) if endswith(lowercase(file), ".h5")]
file = files[ID]
kwargs = NamedTuple()
savename = get_savename(file)
println("$(file) $(savename)")

# Get data
if occursin("simulated", lowercase(file))
    data, parameters, groundtruth = get_data(file; get_groundtruth=true, kwargs...)
else
    data, parameters = get_data(file, kwargs...)
    groundtruth = nothing
end

# Run file
println("Analyzing data")
variables = BindingInference.StateFinder.analyze_trace(
    data[2, :], parameters,
    num_iterations=100,
    saveas=nothing,
    plot=true,
)


