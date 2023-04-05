

function sample_brightness_and_macrostates(data, variables; kwargs...)
    """
    This function jointly samples brightnesses and macrostates in order to allow
    for better mixing. Otherwise we would get stuck in local maximuma where there
    could be twice as many fluorophores with half the brightness.
    """
    
    # For speed, we do not always sample brightnesses and macrostates jointly
    if rand() > .33
        print("0")
        return variables
    end

    # Set up variables
    variables = SimpleNamespace(
        merge(Dict(variables), Dict([(string(key), val) for (key, val) in pairs(kwargs)]))
    )
    mu_micro = variables.mu_micro
    id = findall(mu_micro .!= 0)[1]

    # Propose new brightness
    mu_micro_old = copy(mu_micro)
    mu_micro_new = copy(mu_micro_old) .* rand(Exponential(1))
    variables_old = variables
    variables_new = sample_macrostates(data, variables, mu_micro=mu_micro_new)

    # calculate acceptance probability
    accept_prob = (
        calculate_posterior(data, variables_new)
        - calculate_posterior(data, variables_old)
        + logpdf(Exponential(abs(mu_micro_new[id])), abs(mu_micro_old[id]))
        - logpdf(Exponential(abs(mu_micro_old[id])), abs(mu_micro_new[id]))
    )
    if accept_prob > log(rand())
        variables = variables_new
        print('+')
    else
        variables = variables_old
        print('-')
    end

    # return variables
    return variables

end # function sample_brightness_and_macrostates
