
function sample_transitions(data, variables; kwargs...)

    # Set up variables
    variables = SimpleNamespace(
        merge(Dict(variables), Dict([(string(key), val) for (key, val) in pairs(kwargs)]))
    )

    # Get constants
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
    mu_flor_prop_shape = variables.mu_flor_prop_shape
    mu_back_prop_shape = variables.mu_back_prop_shape

    # Initialize new variables
    pi = copy(pi)

    # Count transitions
    counts = zeros(Int, num_states+1, num_states)
    s_old = 1
    for n = 2:num_frames
        s_new = states[n]
        counts[s_old, s_new] += 1
        s_old = s_new
    end
    
    # Sample probabilities
    for k in 1:num_states
        ids = pi_concentration[k, :] .> 0
        pi[k, ids] = rand(Dirichlet(pi_concentration[k, ids] + counts[k, ids]))
    end

    # Update variables
    variables.pi = pi

    # Return variables
    return variables
end
