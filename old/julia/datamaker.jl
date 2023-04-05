
# Import packages and set up environment
println("Importing packages")
cd(@__DIR__)
using Revise
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
import PyPlot as plt; plt.pygui(true)

# Add local packages
Pkg.develop(path="../Tools/SimpleNamespaces/")
Pkg.develop(path="../Tools/SampleCaches/")
Pkg.develop(path="./BindingInference/")
Pkg.instantiate()
Pkg.resolve()
using SampleCaches
using SimpleNamespaces
import BindingInference as ba


### VARIABLES ###

# Numbers
num_rois = 1000
num_frames = 8000
num_micro = 3
num_max = 1

# Brightness
mu_back = 100
mu_micro = 100
sigma_back = 10
sigma_micro = 10

# Constants
dt = 1e4
seed = 0
concentration = [1, 10, 100]
laser_power = [1, .5, .25]
concentration = repeat(
    concentration, outer=ceil(Int, num_rois / length(concentration))
)[1:num_rois]
laser_power = repeat(
    laser_power, inner=ceil(Int, num_rois / length(laser_power))
)[1:num_rois]

# Binding rates
bind_rate = 1/(dt*num_frames)
unbind_rate = 100/(dt*num_frames)
k_bind = [
    [-unbind_rate, unbind_rate]';
    [bind_rate, -bind_rate]';
    [0, 1]';
]

# Photostate rates
bleach_rate = 1 / (dt * num_frames)
blink_rate = .1 / (dt * num_frames)
unblink_rate = 100 / (dt * num_frames)
switch_rate = 1 / (dt * num_frames)
k_micro = zeros(Float64, num_micro+1, num_micro)
if num_micro == 2
    k_micro[end, :] .= [1, 0]
    k_micro[1, :] .= [-bleach_rate, bleach_rate]
else
    k_micro[end, :] .= [0, fill(1/(num_micro-2), num_micro-2)..., 0]
    k_micro[1, :] .= [
        -unblink_rate, 
        fill(unblink_rate/(num_micro-2), num_micro-2)..., 
        0
    ]
    for k = 2:num_micro-1
        k_micro[k, :] .= [
            blink_rate, 
            fill(switch_rate/(num_micro-2), num_micro-2)..., 
            bleach_rate
        ]
        k_micro[k, k] -= sum(k_micro[k, :])
    end
end

# Create parameters
parameters = Dict([
    # Rates
    ("k_micro", k_micro),              # (1/ns) Photostate transition rates matrix
    ("k_bind", k_bind),                # (1/ns) Binding transition rates matrix
    # Brightness
    ("mu_back", mu_back),              # (ADU)  Brightness of background
    ("mu_micro", mu_micro),            # (ADU)  Brightness of fluorophore microstates
    ("sigma_micro", sigma_micro),      # (ADU)  Photon noise
    ("sigma_back", sigma_back),        # (ADU)  Background noise
    # Constants
    ("seed", seed),                    # (#)    Seed for RNG
    ("dt", dt),                        # (ns)   Time step
    ("concentration", concentration),  # (pM)   Concentrations of binding agent
    ("laser_power", laser_power),      # (mW)   Laser powers
    # Numbers
    ("num_rois", num_rois),            # (#)    Number of ROIs
    ("num_frames", num_frames),        # (#)    Number of time levels
    ("num_micro", num_micro),          # (#)    Number of micro states
    ("num_max", num_max),              # (#)    Maximum number of fluorophores
])


### CREATE DATA ###

# Simulate data
println("Simulating data")
data, groundtruth = ba.simulate_data(parameters=parameters)

# Plot data
println("Plotting data")
ba.plot_variables(data, groundtruth)

# Save data
println("Saving data")
path="/Users/jbryaniv/Desktop/Data/Binding/"
h5 = h5open(path*"simulated_data.h5", "w")
h5["data"] = collect(data')
for (key, value) in parameters
    h5[key] = value
end
gt = create_group(h5, "groundtruth")
for (key, value) in Dict(groundtruth)
    println(key)
    try
        gt[string(key)] = value
    catch
        gt[string(key)] = repr(value)
    end
end
close(h5)


