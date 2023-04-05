
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
        file; path="/Users/jbryaniv/Desktop/Data/Binding/",
        num_iterations=20, plot=true,
        runmode="None", num_rois=nothing, kwargs...
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
        if "laser_power" in keys(parameters)
            parameters["laser_power"] = parameters["laser_power"][ids]
        end
        if "concentration" in keys(parameters)
            parameters["concentration"] = parameters["concentration"][ids]
        end
        if "surface_number" in keys(parameters)
            parameters["surface_number"] = parameters["surface_number"][ids]
        end
    end
    data = data[:, 1:5:end]

    # Set parameters
    parameters = merge(
        parameters,
        Dict([(string(key),value) for (key, value) in pairs(kwargs)])
    )
    if lowercase(runmode) == "pbsa"
        if "g" in keys(parameters)
            g = parameters["g"]
        else
            g = .1
        end
        k_bind = [[0, 0]'; [0, 0]'; [g, 1-g]';]
        k_bind_scale = 0
        background_times = zeros(Bool, size(data))
        background_times[:, round(Int, .9*end):end] .= true
        parameters["g"] = g
        parameters["k_bind"] = k_bind
        parameters["k_bind_scale"] = k_bind_scale
        parameters["background_times"] = background_times
    end

    return data, parameters
end

#### RUN SCRIPT ####

# Get system arg
ID = 2
if length(ARGS) > 0
    ID = parse(Int, ARGS[1]) + 1
end

# Get files and run parameters
path = "/Users/jbryaniv/Desktop/Data/Binding/"
files = [file for file in readdir(path) if endswith(lowercase(file), ".h5")]
files = [file for file in files if !contains(file, "INFERENCE")]
runparams = [(num_rois=10, cameramodel="scmos")]

# Run file and params
file, params = [(file, params) for file in files for params in runparams][ID]
savename = get_savename(file; params...)
println("$(file) $(savename)")
data, parameters = get_data(file; params...)
data, paramters = ba.simulate_data()

# Plot data
fig, ax = plt.subplots(1, 5, sharey=true, sharex=true)
plt.ion()
plt.show()
fig.suptitle("$(file[9:end-3])")
for i in 1:size(ax, 1)
    ax[i].plot(data[i, :])
    ax[i].set_title("ROI $i")
    ax[i].set_xlabel("Time")
end
ax[1].set_ylabel("Intensity")

# Run file
println("Analyzing data")
# variables = SimpleNamespace(merge(ba.PARAMETERS, parameters))
variables =  ba.analyze(
    data, parameters,
    num_iterations=20,
    saveas="outfiles/"*savename,
    plot=false,
)
ba.plot_variables(data, variables)

# Extract results
dt = variables.dt
gain = variables.gain
partitions = variables.partitions
degenerate_ids = variables.degenerate_ids
mu_photo = variables.mu_photo
mu_back = variables.mu_back
sigma_photo = variables.sigma_photo
sigma_back = variables.sigma_back
k_photo = variables.k_photo
k_bind = variables.k_bind
concentration = variables.concentration
laser_power = variables.laser_power
num_rois = variables.num_rois
num_data = variables.num_data
num_photo = variables.num_photo
num_macro = variables.num_macro
num_unique = variables.num_unique
background_times = variables.background_times
cameramodel = variables.cameramodel
macrostates = variables.macrostates
num_bound = variables.num_bound

# ### 1nM ###
# k_photo
#     -0.0035979     0.0035979   0.0
#     0.00109777   -0.00340905  0.00231128
#     0.0           0.0         0.0
#     0.000611619   0.999388    0.0
# k_bind
#     -0.0227829   0.0227829
#     0.0126925  -0.0126925
#     0.0151928   0.984807
# ### 5nM ###
# k_bind
#     -0.0227829   0.0227829
#     0.0126925  -0.0126925
#     0.0211032   0.978897
# k_photo
#     -0.0035979     0.0035979   0.0
#     0.00109777   -0.00340905  0.00231128
#     0.0           0.0         0.0
#     0.000769265   0.999231    0.0
### 10nM ###
# k_photo
#     -0.00325687   0.00325687  0.0
#     0.00109777  -0.00340905  0.00231128
#     0.0          0.0         0.0
#     0.00068189   0.999318    0.0
# k_bind
#     -0.0227829   0.0227829
#     0.0126925  -0.0126925
#     0.0151928   0.984807


Br = 1
D = .01
R = .01
function B(z; R=1, D=1, Br=1)
    value = Br/2*(
        (D+z)/sqrt(R^2+(D+z)^2)
        - z/sqrt(R^2+z^2)
    )
    return value
end

