
function simulate_data(;parameters=Dict(), verbose=true, kwargs...)
    """
    This function simulates the data.
    """

    # Set up parameters
    verbose ? println("Setting up parameters") : nothing
    default_parameters = Dict([
        # Variables
        ("k_photo", nothing),              # (1/ns) Photostate transition rates matrix
        ("mu_back", 100),                  # (ADU)  Brightness of background
        ("mu_photo", 100),                 # (ADU)  Brightness of fluorophore microstates
        ("sigma_photo", 5),                # (ADU)  Photon noise
        ("sigma_back", 5),                 # (ADU)  Background noise
        # Constants
        ("seed", 0),                       # (#)    Seed for RNG
        ("dt", 1e4),                       # (ns)   Time step
        # Numbers
        ("num_rois", 10),                  # (#)    Number of ROIs
        ("num_frames", 1000),              # (#)    Number of time levels
        ("num_photo", 2),                  # (#)    Number of micro states
        ("num_max", 1),                    # (#)    Maximum number of fluorophores
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
    k_photo = variables.k_photo
    mu_back = variables.mu_back
    mu_photo = variables.mu_photo
    sigma_photo = variables.sigma_photo
    sigma_back = variables.sigma_back
    partitions = variables.partitions
    num_rois = variables.num_rois
    num_frames = variables.num_frames
    num_photo = variables.num_photo
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
        mu_macro = partitions * mu_photo .+ mu_back[r]
        sigma_macro = partitions * sigma_photo .+ sigma_back[r]
        pi_macro = micro_rates_to_macro_transitions(
            dt, k_photo, partitions=partitions
        )

        # Sample state trajectory
        s = num_macro + 1
        for n in 1:num_frames
            s = rand(Categorical(pi_macro[s, :]))
            macrostates[r, n] = s
            num_bound[r] = maximum((num_bound[r], num_max-partitions[s, end]))
            data[r, n] = rand(Normal(mu_macro[s], sigma_macro[s]))
        end
    end
    variables.macrostates = macrostates
    variables.num_bound = num_bound

    # Return data
    return data, variables

end # function simulate_data

