module StateFinder

# Export functions
export analyze_trace

# Import packages
using Printf
using HDF5
using Random
using Distributions
import PyPlot as plt; plt.pygui(true)

# Import custom packages
using SimpleNamespaces
using SampleCaches

# Import src files
for file in readdir(joinpath(@__DIR__, "StateFinder/"))
    if endswith(file, ".jl")
        include(joinpath(@__DIR__, "StateFinder/", file))
    end
end

end