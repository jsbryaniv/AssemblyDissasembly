
function sample_gain(data, variables; kwargs...)

    # Set up variables
    variables = SimpleNamespace(
        merge(Dict(variables), Dict([(string(key), val) for (key, val) in pairs(kwargs)]))
    )

    # Get variables
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
    
    # Set up probability
    function probability(gain_)
        prob = logpdf.(Gamma.(gain_shape, gain_scale), gain_)
        for n = 1:num_frames
            scale = (mu_flor_*(states[n]-1) + mu_back_)
            prob += logpdf(Gamma(gain_, scale), data[n])
        end
        return prob
    end

    # Sample multiple times
    for _ in 1:10
        # Sample gain
        gain_old = gain
        gain_new = rand(Gamma(gain_prop_shape, gain_old/gain_prop_shape))
        P_old = probability(gain_old)
        P_new = probability(gain_new)
        acc_prob = (
            P_new - P_old
            + logpdf(Gamma(gain_prop_shape, gain_new/gain_prop_shape), gain_old)
            - logpdf(Gamma(gain_prop_shape, gain_old/gain_prop_shape), gain_new)
        )
        if acc_prob > log(rand())
            gain = gain_new
        end
    end

    # Update variables
    variables.gain = gain

    # Return variables
    return variables
end
