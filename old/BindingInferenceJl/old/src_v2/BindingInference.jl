module BindingInference

# Export functions
export analyze, initialize_variables, simulate_data, plot_variables

# Import packages
using Printf
using HDF5
using Random
using Distributed
using Distributions
using LinearAlgebra
using Statistics
using SpecialFunctions
using LinearMaps
using FillArrays
using SharedArrays
using SparseArrays
using DistributedArrays
import PyPlot as plt; plt.pygui(true)

# Import custom packages
using SimpleNamespaces
using SampleCaches
using PrecomputedCaches

# Import src files
include("parameters.jl")
include("partition_functions.jl")
include("calculate_posterior.jl")
include("initialize_variables.jl")
include("sample_macrostates.jl")
include("sample_brightness.jl")
include("sample_noise.jl")
include("sample_rates.jl")
include("sample_brightness_and_macrostates.jl")
include("simulate_data.jl")
include("plot_variables.jl")
include("analyzer.jl")

# Load packages to workers
@everywhere workers() using Distributed
@everywhere workers() using Distributions
@everywhere workers() using DistributedArrays
@everywhere workers() using SimpleNamespaces
@everywhere workers() using PrecomputedCaches
@everywhere workers() using DistributedArrays
@everywhere workers() include("partition_functions.jl")
@everywhere workers() include("simulate_data.jl")
@everywhere workers() include("initialize_variables.jl")
@everywhere workers() include("sample_macrostates.jl")

end