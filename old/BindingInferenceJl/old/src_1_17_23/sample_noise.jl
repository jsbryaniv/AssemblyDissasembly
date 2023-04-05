
function sample_noise_SCMOS(data, variables; kwargs...)

    # Set up variables
    variables = SimpleNamespace(
        merge(Dict(variables), Dict([(string(key), val) for (key, val) in pairs(kwargs)]))
    )

    # Get variables
    dt = variables.dt
    gain = variables.gain
    mu_back = variables.mu_back
    mu_back_mean = variables.mu_back_mean
    mu_back_vars = variables.mu_back_vars
    mu_photo = variables.mu_photo
    mu_photo_mean = variables.mu_photo_mean
    mu_photo_vars = variables.mu_photo_vars
    sigma_photo = variables.sigma_photo
    sigma_photo_shape = variables.sigma_photo_shape
    sigma_photo_scale = variables.sigma_photo_scale
    sigma_back = variables.sigma_back
    sigma_back_shape = variables.sigma_back_shape
    sigma_back_scale = variables.sigma_back_scale
    k_photo = variables.k_photo
    k_photo_shape = variables.k_photo_shape
    k_photo_scale = variables.k_photo_scale
    k_bind = variables.k_bind
    k_bind_shape = variables.k_bind_shape
    k_bind_scale = variables.k_bind_scale
    macrostates = variables.macrostates
    partitions = variables.partitions
    concentration = variables.concentration
    laser_power = variables.laser_power
    num_rois = variables.num_rois
    num_data = variables.num_data
    num_photo = variables.num_photo
    num_macro = variables.num_macro
    cameramodel = variables.cameramodel
    alphap = variables.sigma_photo_prop_shape
    alphab = variables.sigma_back_prop_shape

    # Reshape populations and add background
    pops = zeros(Float64, num_photo + num_rois, num_rois * num_data)
    dataflat = zeros(Float64, num_rois*num_data)
    for r in 1:num_rois
        for n in 1:num_data
            s = macrostates[r, n]
            pops[1:num_photo, n+num_data*(r-1)] .= partitions[s, :] .* laser_power[r]
            pops[num_photo+r, n+num_data*(r-1)] = 1
            dataflat[n+num_data*(r-1)] = data[r, n]
        end
    end

    # Calculate values
    ids = findall(mu_photo_mean .> 0)  # bright state IDs
    mu = vec(pops' * [mu_photo..., mu_back...])

    # Create conditional posterior function
    function probability(sigma_photo_, sigma_back_)
        sigma = vec(pops' * [sigma_photo_..., sigma_back_...])
        prob = (
            sum(logpdf.(Normal.(mu, sigma), dataflat))
            + sum(logpdf.(Gamma.(sigma_back_shape, sigma_back_scale), sigma_back_))
            + sum(logpdf.(Gamma.(sigma_photo_shape[ids], sigma_photo_scale[ids]), sigma_photo_[ids]))
        )
        return prob
    end

    # Loop through sigma_photo
    for k in 1:num_photo
        sigma_photo_scale[k] == 0 ? continue : nothing
        # Sample multiple times using Metropolis Hastings
        for _ in 1:10
            sigma_photo_old = copy(sigma_photo)
            sigma_photo_new = copy(sigma_photo_old)
            sigma_photo_new[k] = rand(Gamma(alphap, sigma_photo_scale[k]/alphap))
            acc_prob = (
                probability(sigma_photo_new, sigma_back)
                - probability(sigma_photo_old, sigma_back)
                + logpdf(Gamma(alphap, sigma_photo_new[k]/alphap), sigma_photo_old[k])
                - logpdf(Gamma(alphap, sigma_photo_old[k]/alphap), sigma_photo_new[k])
            )
            if acc_prob > log(rand())
                sigma_photo[k] = sigma_photo_new[k]
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
                probability(sigma_photo, sigma_back_new)
                - probability(sigma_photo, sigma_back_old)
                + logpdf(Gamma(alphab, sigma_back_new[r]/alphab), sigma_back_old[r])
                - logpdf(Gamma(alphab, sigma_back_old[r]/alphab), sigma_back_new[r])
            )
            if acc_prob > log(rand())
                sigma_back[r] = sigma_back_new[r]
            end
        end
    end

    # Update variables
    variables.sigma_photo = sigma_photo
    variables.sigma_back = sigma_back

    # Return variables
    return variables
end # sample_noise_SCMOS

