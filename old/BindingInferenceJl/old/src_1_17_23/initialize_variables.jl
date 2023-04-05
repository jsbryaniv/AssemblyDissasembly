
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
    dt = variables.dt
    gain = variables.gain
    concentration = variables.concentration
    laser_power = variables.laser_power
    macrostates = variables.macrostates
    mu_back = variables.mu_back
    mu_back_shape = variables.mu_back_shape
    mu_back_scale = variables.mu_back_scale
    mu_back_mean = variables.mu_back_mean
    mu_back_vars = variables.mu_back_vars
    mu_photo = variables.mu_photo
    mu_photo_shape = variables.mu_photo_shape
    mu_photo_scale = variables.mu_photo_scale
    mu_photo_mean = variables.mu_photo_mean
    mu_photo_vars = variables.mu_photo_vars
    sigma_photo = variables.sigma_photo
    sigma_photo_shape = variables.sigma_photo_shape
    sigma_photo_scale = variables.sigma_photo_scale
    sigma_back = variables.sigma_back
    sigma_back_shape = variables.sigma_back_shape
    sigma_back_scale = variables.sigma_back_scale
    k_bind = variables.k_bind
    k_bind_shape = variables.k_bind_shape
    k_bind_scale = variables.k_bind_scale
    k_photo = variables.k_photo
    k_photo_shape = variables.k_photo_shape
    k_photo_scale = variables.k_photo_scale
    num_rois = variables.num_rois
    num_data = variables.num_data
    num_photo = variables.num_photo
    num_macro = variables.num_macro
    num_max = variables.num_max
    seed = variables.seed
    flor_brightness_guess = variables.flor_brightness_guess
    background_times = variables.background_times
    cameramodel = variables.cameramodel

    # Set RNG seed
    verbose ? println("Setting RNG seed.") : nothing
    Random.seed!(seed)

    # Print status
    verbose ? println("Initializing variables:") : nothing

    # Set up data shapes
    verbose ? println("--Data shape") : nothing
    if data !== nothing
        num_rois, num_data = size(data)
    end
    variables.num_rois = num_rois
    variables.num_data = num_data

    # Check for camera model
    verbose ? println("--Camera model") : nothing
    cameramodel = lowercase(cameramodel)
    if cameramodel == "gaussian"
        cameramodel = "scmos"
    elseif cameramodel == "normal"
        cameramodel = "scmos"
    elseif cameramodel == "gamma"
        cameramodel = "emccd"
    end
    if cameramodel == "emccd"
        if gain === nothing
            error("Gain must be supplied as parameter when using EMCCD noise model.")
        end
    end

    # Background times
    if background_times === nothing
        background_times = zeros(Bool, num_rois, num_data)
    end
    variables.background_times = background_times

    # Concentration and laser power
    verbose ? println("--Conentration and Laser power") : nothing
    if data === nothing
        concentration = repeat(
            concentration, outer=ceil(Int, num_rois / length(concentration))
        )[1:num_rois]
        laser_power = repeat(
            laser_power, inner=ceil(Int, num_rois / length(laser_power))
        )[1:num_rois]
    end
    concentration = concentration .* ones(num_rois)
    laser_power = laser_power .* ones(num_rois)
    variables.concentration = concentration
    variables.laser_power = laser_power

    # Background brightness
    verbose ? println("--Background brightness") : nothing
    if mu_back === nothing
        mu_back = zeros(Float64, num_rois)
        for r in 1:num_rois
            if any(background_times[r, :] .> 0)
                # Estimate background from background times if specified
                mu_back[r] = mean(data[r, findall(background_times[r, :])])
            else
                # Estimate background brightness as the dimmest 10% of the data for each ROI
                mu_back[r] = mean(sort(data[r, :])[1:Int(end÷10)])
            end
        end
        if cameramodel == "emccd"
            mu_back ./= gain
        end
    end
    mu_back = mu_back .* ones(num_rois)
    if cameramodel == "emccd"
        mu_back_shape = mu_back_shape .* ones(Float64, num_rois)
        if mu_back_scale === nothing
            mu_back_scale = mu_back ./ mu_back_shape
        end
    elseif cameramodel == "scmos"
        if mu_back_mean === nothing
            mu_back_mean = mu_back .* ones(Float64, num_rois)
        end
        if mu_back_vars === nothing
            mu_back_vars = var(data) .* ones(Float64, num_rois)
        end
    end
    variables.mu_back = mu_back
    variables.mu_back_shape = mu_back_shape
    variables.mu_back_scale = mu_back_scale
    variables.mu_back_mean = mu_back_mean
    variables.mu_back_vars = mu_back_vars

    # Photo state brightnesses
    verbose ? println("--Photo state brightnesses") : nothing
    if mu_photo === nothing
        if flor_brightness_guess === nothing
            if cameramodel == "emccd"
                flor_brightness_guess = .1 * mean((maximum(data, dims=2) ./ gain - mu_back) ./ laser_power)
            elseif cameramodel == "scmos"
                flor_brightness_guess = .1 * mean(maximum(data, dims=2) - mu_back)
            end
        end
        mu_photo = flor_brightness_guess
    end
    if isa(mu_photo, Number)
        mu_photo = [
            num_photo < 3 ? mu_photo : 0,
            mu_photo .* LinRange(1, 1-.25*(num_photo>3), num_photo - 2)...,
            0
        ]
    end
    if cameramodel == "emccd"
        mu_photo_shape = mu_photo_shape .* ones(Float64, num_photo)
        if mu_photo_scale === nothing
            mu_photo_scale = mu_photo ./ mu_photo_shape
        end
        mu_photo_mean = mu_photo_shape .* mu_photo_scale
    elseif cameramodel == "scmos"
        if mu_photo_mean === nothing
            mu_photo_mean = mu_photo .* ones(Float64, num_photo)
        end
        if mu_photo_vars === nothing
            mu_photo_vars = var(data) .* ones(Float64, num_photo)
        end
    end
    variables.mu_photo = mu_photo
    variables.mu_photo_shape = mu_photo_shape
    variables.mu_photo_scale = mu_photo_scale
    variables.mu_photo_mean = mu_photo_mean
    variables.mu_photo_vars = mu_photo_vars

    # Brightness variance
    if cameramodel == "scmos"
        verbose ? println("--Brightness variance") : nothing
        if sigma_back === nothing
            sigma_back = zeros(Float64, num_rois)
            for r in 1:num_rois
                if any(background_times[r, :] .> 0)
                    # Estimate background from background times if specified
                    sigma_back[r] = std(data[r, findall(background_times[r, :])])
                else
                    # Estimate background as the dimmest 10% of the data for each ROI
                    sigma_back[r] = std(sort(data[r, :])[1:Int(end÷10)])
                end
            end
        end
        sigma_back = sigma_back .* ones(Float64, num_rois)
        sigma_back_shape = sigma_back_shape .* ones(Float64, num_rois)
        if sigma_back_scale === nothing
            sigma_back_scale = sigma_back ./ sigma_back_shape
        end
        if sigma_photo === nothing
            sigma_photo = sqrt(var(data) - .1*mean(sigma_back).^2)
        end
        sigma_photo = sigma_photo .* (mu_photo .> 0)
        sigma_photo_shape = sigma_photo_shape .* ones(Float64, num_photo)
        if sigma_photo_scale === nothing
            sigma_photo_scale = sigma_photo ./ sigma_photo_shape
        end
    end
    variables.sigma_photo = sigma_photo
    variables.sigma_photo_shape = sigma_photo_shape
    variables.sigma_photo_scale = sigma_photo_scale
    variables.sigma_back = sigma_back
    variables.sigma_back_shape = sigma_back_shape
    variables.sigma_back_scale = sigma_back_scale

    # Binding rates
    verbose ? println("--Binding rates") : nothing
    k_bind_shape = k_bind_shape .* ones(Float64, 3, 2)
    if k_bind === nothing
        # Set some bind rates
        bind_rate = 10/(dt*num_data)
        unbind_rate = 10/(dt*num_data)
        k_bind = [
            [-unbind_rate, unbind_rate]';
            [bind_rate, -bind_rate]';
            [1/num_max, 1-1/num_max]';
        ]
    end
    if k_bind_scale === nothing
        k_bind_scale = copy(k_bind)
        k_bind_scale[1:end-1, :] ./ k_bind_shape[1:end-1, :]
    end
    variables.k_bind = k_bind
    variables.k_bind_shape = k_bind_shape
    variables.k_bind_scale = k_bind_scale

    # Photostate transition rates
    verbose ? println("--Photostate transition rates") : nothing
    k_photo_shape = k_photo_shape .* ones(Float64, num_photo + 1, num_photo)
    if k_photo === nothing
        # Set some microphysical rates
        bleach_rate = (.1 * num_max) / (dt * num_data)
        blink_rate = (.1 * num_max) / (dt * num_data)
        unblink_rate = (.1 * num_max) / (dt * num_data)
        switch_rate = (.1 * num_max) / (dt * num_data)
        k_photo = zeros(Float64, num_photo+1, num_photo)
        # Initial state probability
        k_photo[end, :] .= [fill(1/(num_photo-1), num_photo-1)..., 0]
        # Photostate transition rates
        if num_photo == 2
            k_photo[1, :] .= [-bleach_rate, bleach_rate]
        else
            k_photo[1, :] .= [
                -unblink_rate, 
                fill(unblink_rate/(num_photo-2), num_photo-2)..., 
                0
            ]
            for k = 2:num_photo-1
                k_photo[k, :] .= [
                    blink_rate, 
                    fill(switch_rate/(num_photo-2), num_photo-2)..., 
                    bleach_rate
                ]
                k_photo[k, k] -= sum(k_photo[k, :])
            end
        end
    end
    if k_photo_scale === nothing
        k_photo_scale = copy(k_photo)
        k_photo_scale[1:end-1, :] ./= k_photo_shape[1:end-1, :]
    end
    variables.k_photo = k_photo
    variables.k_photo_shape = k_photo_shape
    variables.k_photo_scale = k_photo_scale

    # State partitions
    verbose ? println("--State partitions") : nothing
    partitions = calculate_partitions(num_max, num_photo+1)[:, 1:end-1]
    num_macro = size(partitions, 1)
    variables.partitions = partitions
    variables.num_macro = num_macro

    # Macrostate trajectory
    verbose ? println("--Macrostate trajectory") : nothing  
    if macrostates === nothing
        macrostates = num_macro * ones(Int, num_rois, num_data)
    end
    variables.macrostates = macrostates

    # Number of bound fluorophores
    verbose ? println("--Number of bound fluorophores") : nothing
    num_bound = zeros(Int, num_rois)
    for r in 1:num_rois
        for n in 1:num_data
            s = macrostates[r, n]
            M = sum(partitions[s, :])
            M > num_bound[r] ? num_bound[r] = M : nothing
        end
    end
    variables.num_bound = num_bound

    # Find degenerate macrostate brightnesses
    verbose ? println("--Degenerate macrostates") : nothing
    mu_macro = partitions * mu_photo
    unique_mu = unique(mu_macro)
    num_unique = length(unique_mu)
    degenerate_ids = Vector{Vector{Int}}(undef, num_unique)
    for u = 1:num_unique
        ids = findall(mu_macro .== unique_mu[u])
        degenerate_ids[u] = ids
    end
    variables.num_unique = num_unique
    variables.degenerate_ids = degenerate_ids
    
    # Probability
    verbose ? println("--Probability") : nothing
    P = -Inf
    variables.P = P

    # Print status
    verbose ? println("Variables initialized") : nothing

    # Return variables
    return variables
end # function initialize_variables

