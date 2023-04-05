
function simulate_data(;parameters=Dict(), verbose=true, kwargs...)
    """
    This function simulates the data.
    """

    # Set up parameters
    verbose ? println("Setting up parameters") : nothing
    default_parameters = Dict([
        # Variables
        ("k_micro", nothing),              # (1/ns) Photostate transition rates matrix
        ("k_bind", nothing),               # (1/ns) Binding transition rates matrix
        ("mu_back", 100),                  # (ADU)  Brightness of background
        ("mu_micro", 100),                 # (ADU)  Brightness of fluorophore microstates
        ("sigma_micro", 10),               # (ADU)  Photon noise
        ("sigma_back", 100),               # (ADU)  Background noise
        # Constants
        ("seed", 0),                       # (#)    Seed for RNG
        ("dt", 1e4),                       # (ns)   Time step
        ("concentration", [1, 10, 100]),   # (pM)   Concentrations of binding agent
        ("laser_power", [1, .5, .25]),     # (mW)   Laser powers
        # Numbers
        ("num_rois", 10),                  # (#)    Number of ROIs
        ("num_frames", 2000),              # (#)    Number of time levels
        ("num_micro", 3),                  # (#)    Number of micro states
        ("num_max", 20),                   # (#)    Maximum number of fluorophores
    ])
    parameters = merge(
        default_parameters, 
        parameters, 
        Dict([(string(key), val) for (key, val) in kwargs]),
    )

    # Initialize variables
    verbose ? println("Setting up variables") : nothing
    variables = initialize_variables(nothing, parameters, verbose=verbose)

    # Extract variables
    verbose ? println("Extracting variables") : nothing
    seed = variables.seed
    dt = variables.dt
    k_micro = variables.k_micro
    k_bind = variables.k_bind
    mu_back = variables.mu_back
    mu_micro = variables.mu_micro
    sigma_micro = variables.sigma_micro
    sigma_back = variables.sigma_back
    partitions = variables.partitions
    concentration = variables.concentration
    laser_power = variables.laser_power
    num_rois = variables.num_rois
    num_frames = variables.num_frames
    num_micro = variables.num_micro
    num_max = variables.num_max
    num_macro = variables.num_macro
    
    # set RNG seed
    verbose ? println("Setting RNG seed") : nothing
    Random.seed!(seed)

    # Simulate data
    verbose ? println("Simulating data") : nothing
    data = zeros(num_rois, num_frames)
    macrostates = zeros(Int, num_rois, num_frames)
    num_bound = zeros(Int, num_rois)
    for r in 1:num_rois
        verbose ? println("--ROI $(r)") : nothing
        
        # Caclulate macrostate features
        mu_macro = (partitions * mu_micro) .* laser_power[r] .+ mu_back[r]
        sigma_macro = (partitions * sigma_micro) .* laser_power[r] .+ sigma_back[r]
        pi_macro = micro_rates_to_macro_transitions(
            dt, k_micro, k_bind, C=concentration[r], W=laser_power[r], partitions=partitions
        )

        # Sample state trajectory
        s = num_macro + 1
        for n in 1:num_frames
            s = rand(Categorical(pi_macro[s, :]/sum(pi_macro[s, :])))
            macrostates[r, n] = s
            data[r, n] = rand(Normal(mu_macro[s], sigma_macro[s]))
            num_bound[r] = maximum((num_bound[r], sum(partitions[s, 1:end-1])))
        end
    end
    variables.macrostates = macrostates
    variables.num_bound = num_bound

    # Return data
    return data, variables

end # function simulate_data

