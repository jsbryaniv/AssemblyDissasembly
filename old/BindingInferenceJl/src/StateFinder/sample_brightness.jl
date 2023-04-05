
function sample_brightness(data, variables; kwargs...)

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
    mu_flor_prop_shape = variables.mu_flor_prop_shape
    mu_back_prop_shape = variables.mu_back_prop_shape
    
    # Set up probability
    function probability(mu_flor_, mu_back_)
        prob = (
            sum(logpdf.(Gamma.(mu_flor_shape, mu_flor_scale), mu_flor_))
            + sum(logpdf.(Gamma.(mu_back_shape, mu_back_scale), mu_back_))
        )
        for n = 1:num_frames
            scale = (mu_flor_*(states[n]-1) + mu_back_)
            prob += logpdf(Gamma(gain, scale), data[n])
        end
        return prob
    end

    # Sample multiple times
    for _ in 1:10

        # Sample mu_flor
        mu_flor_old = mu_flor
        mu_flor_new = rand(Gamma(mu_flor_prop_shape, mu_flor_old/mu_flor_prop_shape))
        P_old = probability(mu_flor_old, mu_back)
        P_new = probability(mu_flor_new, mu_back)
        acc_prob = (
            P_new - P_old
            + logpdf(Gamma(mu_flor_prop_shape, mu_flor_new/mu_flor_prop_shape), mu_flor_old)
            - logpdf(Gamma(mu_flor_prop_shape, mu_flor_old/mu_flor_prop_shape), mu_flor_new)
        )
        if acc_prob > log(rand())
            mu_flor = mu_flor_new
        end

        # Sample mu_back
        mu_back_old = mu_back
        mu_back_new = rand(Gamma(mu_back_prop_shape, mu_back_old/mu_back_prop_shape))
        P_old = probability(mu_flor, mu_back_old)
        P_new = probability(mu_flor, mu_back_new)
        acc_prob = (
            P_new - P_old
            + logpdf(Gamma(mu_back_prop_shape, mu_back_new/mu_back_prop_shape), mu_back_old)
            - logpdf(Gamma(mu_back_prop_shape, mu_back_old/mu_back_prop_shape), mu_back_new)
        )
        if acc_prob > log(rand())
            mu_back = mu_back_new
        end
    end

    # Update variables
    variables.mu_flor = mu_flor
    variables.mu_back = mu_back

    # Return variables
    return variables
end
