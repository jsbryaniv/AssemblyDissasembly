
function sample_noise(data, variables; kwargs...)

    # Set up variables
    variables = SimpleNamespace(
        merge(Dict(variables), Dict([(string(key), val) for (key, val) in pairs(kwargs)]))
    )

    # Get variables
    dt = variables.dt
    mu_back = variables.mu_back
    mu_back_mean = variables.mu_back_mean
    mu_back_std = variables.mu_back_std
    mu_micro = variables.mu_micro
    mu_micro_mean = variables.mu_micro_mean
    mu_micro_std = variables.mu_micro_std
    sigma_micro = variables.sigma_micro
    sigma_micro_shape = variables.sigma_micro_shape
    sigma_micro_scale = variables.sigma_micro_scale
    sigma_back = variables.sigma_back
    sigma_back_shape = variables.sigma_back_shape
    sigma_back_scale = variables.sigma_back_scale
    k_micro = variables.k_micro
    k_micro_shape = variables.k_micro_shape
    k_micro_scale = variables.k_micro_scale
    k_bind = variables.k_bind
    k_bind_shape = variables.k_bind_shape
    k_bind_scale = variables.k_bind_scale
    macrostates = variables.macrostates
    partitions = variables.partitions
    concentration = variables.concentration
    laser_power = variables.laser_power
    num_rois = variables.num_rois
    num_frames = variables.num_frames
    num_micro = variables.num_micro
    num_macro = variables.num_macro
    alphap = variables.sigma_micro_prop_shape
    alphab = variables.sigma_back_prop_shape

    # Reshape populations and add background
    pops = zeros(Float64, num_micro + num_rois, num_rois * num_frames)
    dataflat = zeros(Float64, num_rois*num_frames)
    for r in 1:num_rois
        for n in 1:num_frames
            s = macrostates[r, n]
            pops[1:num_micro, n+num_frames*(r-1)] .= partitions[s, :] .* laser_power[r]
            pops[num_micro+r, n+num_frames*(r-1)] = 1
            dataflat[n+num_frames*(r-1)] = data[r, n]
        end
    end

    # Calculate values
    ids = findall(mu_micro_mean .> 0)  # bright state IDs
    mu = vec(pops' * [mu_micro..., mu_back...])

    # Create conditional posterior function
    function probability(sigma_micro_, sigma_back_)
        sigma = vec(pops' * [sigma_micro_..., sigma_back_...])
        prob = (
            sum(logpdf.(Normal.(mu, sigma), dataflat))
            + sum(logpdf.(Gamma.(sigma_back_shape, sigma_back_scale), sigma_back_))
            + sum(logpdf.(Gamma.(sigma_micro_shape[ids], sigma_micro_scale[ids]), sigma_micro_[ids]))
        )
        return prob
    end

    # Loop through sigma_micro
    for k in 1:num_micro
        sigma_micro_scale[k] == 0 ? continue : nothing
        # Sample multiple times using Metropolis Hastings
        for _ in 1:10
            sigma_micro_old = copy(sigma_micro)
            sigma_micro_new = copy(sigma_micro_old)
            sigma_micro_new[k] = rand(Gamma(alphap, sigma_micro_scale[k]/alphap))
            acc_prob = (
                probability(sigma_micro_new, sigma_back)
                - probability(sigma_micro_old, sigma_back)
                + logpdf(Gamma(alphap, sigma_micro_new[k]/alphap), sigma_micro_old[k])
                - logpdf(Gamma(alphap, sigma_micro_old[k]/alphap), sigma_micro_new[k])
            )
            if acc_prob > log(rand())
                sigma_micro[k] = sigma_micro_new[k]
            end
        end
    end

    # Loop through sigma_back
    for r in 1:num_rois
        # Sample multiple times using Metropolis Hastings
        for _ in 1:5
            sigma_back_old = copy(sigma_back)
            sigma_back_new = copy(sigma_back_old)
            sigma_back_new[r] = rand(Gamma(alphab, sigma_back_scale[r]/alphab))
            acc_prob = (
                probability(sigma_micro, sigma_back_new)
                - probability(sigma_micro, sigma_back_old)
                + logpdf(Gamma(alphab, sigma_back_new[r]/alphab), sigma_back_old[r])
                - logpdf(Gamma(alphab, sigma_back_old[r]/alphab), sigma_back_new[r])
            )
            if acc_prob > log(rand())
                sigma_back[r] = sigma_back_new[r]
            end
        end
    end

    # Update variables
    variables.sigma_micro = sigma_micro
    variables.sigma_back = sigma_back

    # Return variables
    return variables
end # sample_noise

