

function sample_brightness_and_states(data, variables; kwargs...)
    """
    This function jointly samples brightnesses and states in order to allow
    for better mixing. Otherwise we would get stuck in local maximuma where there
    could be twice as many fluorophores with half the brightness.
    """
    
    # For speed, we do not always sample brightnesses and states jointly
    if rand() > .33
        print("0")
        return variables
    end

    # Set up variables
    variables = SimpleNamespace(
        merge(Dict(variables), Dict([(string(key), val) for (key, val) in pairs(kwargs)]))
    )
    pi = variables.pi
    mu_flor = variables.mu_flor
    num_states = variables.num_states
    pi_concentration = variables.pi_concentration
    
    # Propose new brightness
    mu_flor_old = mu_flor
    mu_flor_new = mu_flor_old * rand(Exponential(1))
    # pi_old = pi
    # pi_new = zeros(num_states+1, num_states)
    # for k in 1:(num_states+1)
    #     ids = pi_concentration[k, :] .> 0
    #     pi_new[k, ids] = rand(Dirichlet(pi_concentration[k, ids]))
    # end
    variables_old = copy(variables)
    variables_new = sample_states(data, variables, mu_flor=mu_flor_new)
    accept_prob = (
        calculate_posterior(data, variables_new)
        - calculate_posterior(data, variables_old)
        + logpdf(Exponential(mu_flor_new), mu_flor_old)
        - logpdf(Exponential(mu_flor_old), mu_flor_new)
    )
    if accept_prob > log(rand())
        variables = variables_new
        print('+')
    else
        variables = variables_old
        print('-')
    end

    # Return variables
    return variables
end
