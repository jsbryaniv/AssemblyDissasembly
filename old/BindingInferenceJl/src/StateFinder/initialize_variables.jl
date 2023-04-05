
function initialize_variables(data, parameters=Dict(); verbose=false, kwargs...)

    # Set up variables
    verbose ? println("Initializing parameters.") : nothing
    parameters = merge(
        PARAMETERS,
        parameters, 
        Dict([(string(key), val) for (key, val) in pairs(kwargs)])
    )
    variables = SimpleNamespace(parameters)

    # Get constants
    verbose ? println("Extracting constants.") : nothing
    P = variables.P
    pi = variables.pi
    gain = variables.gain
    states = variables.states
    mu_flor = variables.mu_flor
    mu_back = variables.mu_back
    gain_shape = variables.gain_shape
    gain_scale = variables.gain_scale
    mu_flor_shape = variables.mu_flor_shape
    mu_flor_scale = variables.mu_flor_scale
    mu_back_shape = variables.mu_back_shape
    mu_back_scale = variables.mu_back_scale
    pi_concentration = variables.pi_concentration
    num_frames = variables.num_frames
    num_states = variables.num_states
    seed = variables.seed
    gain_prop_shape = variables.gain_prop_shape
    mu_flor_prop_shape = variables.mu_flor_prop_shape
    mu_back_prop_shape = variables.mu_back_prop_shape

    # Set RNG seed
    verbose ? println("Setting RNG seed.") : nothing
    Random.seed!(seed)

    # Print status
    verbose ? println("Initializing variables:") : nothing

    # Set up data shapes
    verbose ? println("--Data shape") : nothing
    if data !== nothing
        num_frames = size(data, 1)
    end
    variables.num_frames = num_frames

    # Gain
    verbose ? println("--Gain") : nothing
    if gain === nothing
        gain = 10
    end
    if gain_scale === nothing
        gain_scale = gain / gain_shape
    end
    variables.gain = gain
    variables.gain_shape = gain_shape
    variables.gain_scale = gain_scale


    # Background brightness
    verbose ? println("--Background brightness") : nothing
    if mu_back === nothing
        # Estimate background brightness as the dimmest 10% of the data for each ROI
        mu_back = mean(sort(data)[1:Int(end÷10)]) / gain
    end
    if mu_back_scale === nothing
        mu_back_scale = mu_back / mu_back_shape
    end
    variables.mu_back = mu_back
    variables.mu_back_shape = mu_back_shape
    variables.mu_back_scale = mu_back_scale

    # Photo state brightnesses
    verbose ? println("--Photo state brightnesses") : nothing
    if mu_flor === nothing
        # Estimate photo state brightnesses as the brightest 10% of the data for each ROI
        mu_flor = mean(sort(data)[Int(end÷10):end]) / gain - mu_back
    end
    if mu_flor_scale === nothing
        mu_flor_scale = mu_flor / mu_flor_shape
    end
    variables.mu_flor = mu_flor
    variables.mu_flor_shape = mu_flor_shape
    variables.mu_flor_scale = mu_flor_scale

    # Transition matrix
    verbose ? println("--Transition matrix") : nothing
    if pi === nothing
        pi = zeros(num_states + 1, num_states)
        for k in 1:num_states
            pi[k, k] = 10
            if k < num_states
                pi[k, k + 1] = 1
            end
            if k > 1
                pi[k, k - 1] = 1
            end
            pi[k, :] /= sum(pi[k, :])
        end
        pi[end, :] = num_states:-1:1
        pi[end, :] /= sum(pi[end, :])
    end
    if pi_concentration === nothing
        pi_concentration = pi
    end
    variables.pi = pi
    variables.pi_concentration = pi_concentration

    # State trajectory
    verbose ? println("--State trajectory") : nothing  
    if states === nothing
        states = ones(Int, num_frames)
    end
    variables.states = states
    
    # Probability
    verbose ? println("--Probability") : nothing
    P = -Inf
    variables.P = P

    # Print status
    verbose ? println("Variables initialized.") : nothing

    # Return variables
    return variables
end # function initialize_variables

