

function sample_brightness_EMCCD(data, variables; kwargs...)

    # Set up variables
    variables = SimpleNamespace(
        merge(Dict(variables), Dict([(string(key), val) for (key, val) in pairs(kwargs)]))
    )

    # Get variables
    dt = variables.dt
    gain = variables.gain
    mu_photo = variables.mu_photo
    mu_photo_shape = variables.mu_photo_shape
    mu_photo_scale = variables.mu_photo_scale
    mu_back = variables.mu_back
    mu_back_shape = variables.mu_back_shape
    mu_back_scale = variables.mu_back_scale
    macrostates = variables.macrostates
    partitions = variables.partitions
    laser_power = variables.laser_power
    num_rois = variables.num_rois
    num_data = variables.num_data
    num_photo = variables.num_photo

    # Initialize variables
    mu_photo = copy(mu_photo)
    mu_back = copy(mu_back)

    # Calculate variables
    num_mus = num_photo + num_rois
    ids = findall(mu_photo_scale .> 0)  # bright state IDs

    # Reshape populations and add background
    pops = zeros(Float64, num_mus, num_rois * num_data)
    dataflat = zeros(Float64, num_rois * num_data)
    for r in 1:num_rois
        for n in 1:num_data
            s = macrostates[r, n]
            pops[1:num_photo, n+num_data*(r-1)] .= partitions[s, :] .* laser_power[r]
            pops[num_photo+r, n+num_data*(r-1)] = 1
            dataflat[n+num_data*(r-1)] = data[r, n]
        end
    end

    # Reshape prior for HMC
    shape = zeros(Float64, num_mus, 1)
    shape[1:num_photo, 1] .= mu_photo_shape
    shape[num_photo+1:end, 1] .= mu_back_shape
    scale = zeros(Float64, num_mus, 1)
    scale[ids, 1] .= mu_photo_scale[ids]
    scale[num_photo+1:end, 1] .= mu_back_scale
    idx = scale .> 0  # ids of bright micro states

    # Set constants for HMC
    h = rand(Exponential(1/(num_rois*num_data)))
    num_steps = rand(Poisson(100))
    mass = 1
    masses = idx .* mass      # mass of dark states is ignored (set to 0)
    masses_inv = idx ./ mass  # inverse mass of bright states is 0

    # Initialize brightness vector and momemtum
    q = zeros(Float64, num_mus, 1)
    q[1:num_photo, 1] .= mu_photo
    q[num_photo+1:end, 1] .= mu_back
    p = rand(Normal(0, 1), num_mus, 1) .* sqrt.(masses)
    q_old = copy(q)
    p_old = copy(p)

    # Conditional probability for q
    function probability(q_, p_)

        # automatically reject if any brightnesses are negative
        if any(q_ .< 0)
            return -Inf
        end

        # calculate probability
        prob = (
            sum(logpdf.(Gamma.(.5*vec(q_'*pops), 2*gain), dataflat))    # likelihood
            + sum(logpdf.(Gamma.(shape[idx], scale[idx]), q_[idx]))  # prior
            + sum(logpdf.(Normal.(0, masses[idx]), p_[idx]))         # momentum
        )

        # return probability
        return prob
    end

    # Gradient of the Hamiltonian
    function dH_dq(q_)

        # set up gradient
        g = zeros(Float64, num_mus, 1)

        # reject sample and stop calculating if any brightnesses are negative
        if any(q_ .< 0)
            return g 
        end

        # calculate gradient
        g[idx] .= (
            ( # prior gradient
                (shape[idx] .- 1) ./ (.5 .* q_[idx]) 
                .- (1 ./ scale[idx])
            )                          
            .+ ( # likelihood gradient
                (
                    .5 .* pops * (
                        log.(dataflat ./ (2 * gain)) 
                        .- vec(digamma.(.5 .* (q_' * pops)))
                    )
                )[findall(idx)] 
            )
        )

        # return gradient
        return g
    end

    # HMC
    for i = 1:num_steps
        p .= p .+ (.5 * h) .* dH_dq(q)
        q .= q .+ h .* p .* masses_inv
        p .= p .+ (.5 * h) .* dH_dq(q)
    end

    # Accept or reject
    P_new = probability(q, p)
    P_old = probability(q_old,  p_old)
    accept_prob = P_new - P_old
    if accept_prob > log(rand())
        print('+')
    else
        q = q_old
        print('-')
    end

    # Update variables
    mu_photo = q[1:num_photo, 1]
    mu_back = q[num_photo+1:end, 1]
    variables.mu_photo = mu_photo
    variables.mu_back = mu_back

    # Return variables
    return variables
end # sample_brightness_EMCCD


function sample_brightness_SCMOS(data, variables; kwargs...)

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
    num_macro = variables.num_macro
    num_photo = variables.num_photo
    cameramodel = variables.cameramodel

    # Calculate variables
    ids = findall(mu_photo_mean .> 0)  # bright state IDs
    num_mus = length(ids) + num_rois

    # Reshape populations and add background
    pops = zeros(Float64, num_mus, num_rois * num_data)
    dataflat = zeros(Float64, num_rois * num_data)
    Sigma_inv = zeros(Float64, num_rois * num_data)
    for r in 1:num_rois
        sigma_macro = (partitions * sigma_photo) .* laser_power[r] .+ sigma_back[r]
        for n in 1:num_data
            s = macrostates[r, n]
            pops[1:end-num_rois, n+num_data*(r-1)] .= partitions[s, ids] .* laser_power[r]
            pops[end-num_rois+r, n+num_data*(r-1)] = 1
            dataflat[n+num_data*(r-1)] = data[r, n]
            Sigma_inv[n+num_data*(r-1)] = 1 / (sigma_macro[s] .^ 2)
        end
    end
    Sigma_inv = spdiagm(Sigma_inv)

    # Set up prior
    mu0 = [mu_photo_mean[ids]..., mu_back_mean...]
    Sigma0_inv = diagm(1 ./ ([mu_photo_vars[ids]..., mu_back_vars...].^2))

    # Sample brightnesses
    cov = Sigma0_inv + pops * Sigma_inv * pops'
    cov = inv(cov) # + .01 * maximum(cov) * Matrix(I, num_mus, num_mus))
    cov = (cov + cov')/2
    mu = cov * (Sigma0_inv * mu0 + pops * Sigma_inv * dataflat)
    q = rand(MvNormal(vec(mu), cov))
    mu_photo[ids] = q[1:length(ids)]
    mu_back = q[length(ids)+1:end]

    # Update variables
    variables.mu_photo = mu_photo
    variables.mu_back = mu_back

    # Return variables
    return variables
end # sample_brightness_SCMOS


function sample_brightness(data, variables; kwargs...)

    # Set up variables
    variables = SimpleNamespace(
        merge(Dict(variables), Dict([(string(key), val) for (key, val) in pairs(kwargs)]))
    )

    # Pick sample from noise model
    cameramodel = variables.cameramodel
    if cameramodel == "emccd"
        variables = sample_brightness_EMCCD(data, variables)
    elseif cameramodel == "scmos"
        variables = sample_brightness_SCMOS(data, variables)
        variables = sample_noise_SCMOS(data, variables)
    end

    # Return variables
    return variables
end # sample_brightness
