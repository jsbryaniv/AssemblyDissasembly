
function calculate_posterior(data, variables; verbose=false, kwargs...)

    # Set up variables
    verbose ? println("Setting up variables") : nothing
    variables = SimpleNamespace(
        merge(Dict(variables), Dict([(string(key), val) for (key, val) in pairs(kwargs)]))
    )

    # Get variables
    verbose ? println("Extracting variables") : nothing
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

    # Calculate prior
    verbose ? println("Calculating prior") : nothing
    prior = (
        sum(logpdf.(Gamma.(gain_shape, gain_scale), gain))
        + sum(logpdf.(Gamma.(mu_flor_shape, mu_flor_scale), mu_flor))
        + sum(logpdf.(Gamma.(mu_back_shape, mu_back_scale), mu_back))
    )
    for k in 1:num_states
        ids = pi_concentration[k, :] .> 0
        prior += logpdf(Dirichlet(pi_concentration[k, ids]), pi[k, ids])
    end
    
    # Calculate kinetic part
    verbose ? println("Calculating dynamics part") : nothing
    kinetic = 0
    s_old = num_states + 1
    for n in 1:num_frames
        s_new = states[n]
        kinetic += log(pi[s_old, s_new])
        s_old = s_new
    end

    # Calculate likelihood
    verbose ? println("Calculating likelihood") : nothing
    lhood = 0
    for n = 1:num_frames
        scale = (mu_flor*(states[n]-1) + mu_back)
        lhood += logpdf(Gamma(gain, scale), data[n])
    end

    # Calculate posterior
    verbose ? println("Calculating posterior") : nothing
    posterior = prior + kinetic + lhood

    # Return posterior
    return posterior
end # calculate_posterior
