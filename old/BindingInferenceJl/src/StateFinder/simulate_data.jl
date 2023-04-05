
function simulate_data(;parameters=Dict(), verbose=true, kwargs...)
    """
    This function simulates the data.
    """

    # Set up parameters
    verbose ? println("Setting up parameters") : nothing
    default_parameters = Dict([
        # Variables
        ("pi", nothing),
        ("gain", 10),
        ("mu_back", 100),                 # (ADU)  Brightness of background
        ("mu_flor", 100),                 # (ADU)  Brightness of fluorophore microstates
        # Constants
        ("seed", 0),                       # (#)    Seed for RNG
        ("dt", 1e4),                       # (ns)   Time step
        # Numbers
        ("num_frames", 2000),              # (#)    Number of time levels
        ("num_states", 20),                # (#)    Maximum number of fluorophores
    ])
    parameters = merge(
        default_parameters, 
        parameters, 
        Dict([(string(key), val) for (key, val) in kwargs]),
    )
    if parameters["pi"] === nothing
        pi = zeros(num_states+1, num_states)
        for k in 1:num_states
            pi[k, k] = 1
            if k < num_states
                pi[k, k+1] = 1
            end
            if k > 1
                pi[k, k-1] = 1
            end
            pi[k, :] /= sum(pi[k, :])
        end
        pi[end, 1] = 1
        parameters["pi"] = pi
    end

    # Initialize variables
    verbose ? println("Setting up variables") : nothing
    variables = initialize_variables(nothing, parameters, verbose=verbose)

    # Extract variables
    verbose ? println("Extracting variables") : nothing
    pi = variables.pi
    gain = variables.gain
    mu_back = variables.mu_back
    mu_flor = variables.mu_flor
    num_frames = variables.num_frames
    num_states = variables.num_states
    seed = variables.seed
    
    # set RNG seed
    verbose ? println("Setting RNG seed") : nothing
    Random.seed!(seed)

    # Simulate data
    verbose ? println("Simulating data") : nothing
    data = zeros(num_frames)
    states = zeros(Int, num_frames)

    # Sample state trajectory
    s = num_states + 1
    for n in 1:num_frames
        s = rand(Categorical(pi[s, :]))
        states[r, n] = s
        data[n] = rand(Gamma(gain, mu_flor*(s-1) + mu_back))
    end

    # Update variables
    variables.states = states

    # Return data
    return data, variables

end # function simulate_data

