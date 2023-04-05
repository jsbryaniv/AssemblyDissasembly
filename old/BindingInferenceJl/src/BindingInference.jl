module BindingInference

# Export functions
export analyze_data

# Import packages
using Printf
using HDF5
using Random
using Distributed
using Distributions
using LinearAlgebra
using Statistics
using SpecialFunctions
using SharedArrays
using SparseArrays
using DistributedArrays
import PyPlot as plt; plt.pygui(true)

# Import custom packages
using SimpleNamespaces
using SampleCaches

# Import src files
for file in readdir(joinpath(@__DIR__, "BindingInference/"))
    if endswith(file, ".jl")
        include(joinpath(@__DIR__, "BindingInference/", file))
    end
end

# Import StateFinder
include(joinpath(@__DIR__, "StateFinder.jl"))
using .StateFinder

end