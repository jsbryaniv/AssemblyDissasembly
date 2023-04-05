
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
    concentration = variables.concentration
    laser_power = variables.laser_power
    macrostates = variables.macrostates
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
    k_bind = variables.k_bind
    k_bind_shape = variables.k_bind_shape
    k_bind_scale = variables.k_bind_scale
    k_micro = variables.k_micro
    k_micro_shape = variables.k_micro_shape
    k_micro_scale = variables.k_micro_scale
    num_rois = variables.num_rois
    num_frames = variables.num_frames
    num_micro = variables.num_micro
    num_macro = variables.num_macro
    num_max = variables.num_max
    seed = variables.seed
    flor_brightness_guess = variables.flor_brightness_guess

    # Set RNG seed
    verbose ? println("Setting RNG seed.") : nothing
    Random.seed!(seed)

    # Print status
    verbose ? println("Initializing variables:") : nothing

    # Set up data shapes
    verbose ? println("--Data shape") : nothing
    if data !== nothing
        num_rois, num_frames = size(data)
    end
    variables.num_rois = num_rois
    variables.num_frames = num_frames

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
            # Estimate background brightness as the dimmest 10% of the data for each ROI
            mu_back[r] = mean(sort(data[r, :])[1:Int(end÷10)])
        end
    end
    mu_back = mu_back .* ones(num_rois)
    if mu_back_mean === nothing
        mu_back_mean = mu_back .* ones(Float64, num_rois)
    end
    if mu_back_std === nothing
        mu_back_std = mu_back .* ones(Float64, num_rois)
    end
    variables.mu_back = mu_back
    variables.mu_back_mean = mu_back_mean
    variables.mu_back_std = mu_back_std

    # Photo state brightnesses
    verbose ? println("--Photo state brightnesses") : nothing
    if mu_micro === nothing
        if flor_brightness_guess === nothing
            flor_brightness_guess = mean(.9 * (maximum(data, dims=2) - mu_back) ./ laser_power)
        end
        mu_micro = flor_brightness_guess
    end
    if isa(mu_micro, Number)
        mu_micro = [
            num_micro < 3 ? mu_micro : 0,
            mu_micro .* LinRange(1, 1-.25*(num_micro>3), num_micro - 2)...,
            0
        ]
    end
    if mu_micro_mean === nothing
        mu_micro_mean = mu_micro .* ones(Float64, num_micro)
    end
    if mu_micro_std === nothing
        mu_micro_std = mu_micro .* ones(Float64, num_micro)
    end
    variables.mu_micro = mu_micro
    variables.mu_micro_mean = mu_micro_mean
    variables.mu_micro_std = mu_micro_std

    # Brightness variance
    verbose ? println("--Brightness variance") : nothing
    if sigma_back === nothing
        sigma_back = zeros(Float64, num_rois)
        for r in 1:num_rois
            # Estimate background as the dimmest 50% of the data for each ROI
            sigma_back[r] = std(sort(data[r, :])[1:Int(end÷2)])
        end
    end
    sigma_back = sigma_back .* ones(Float64, num_rois)
    sigma_back_shape = sigma_back_shape .* ones(Float64, num_rois)
    if sigma_back_scale === nothing
        sigma_back_scale = sigma_back ./ sigma_back_shape
    end
    if sigma_micro === nothing
        sigma_micro = mean([std(sort(data[r, :])[Int(end÷10):end]) for r in 1:num_rois])
    end
    sigma_micro = sigma_micro .* (mu_micro .> 0)
    sigma_micro_shape = sigma_micro_shape .* ones(Float64, num_micro)
    if sigma_micro_scale === nothing
        sigma_micro_scale = sigma_micro ./ sigma_micro_shape
    end
    variables.sigma_micro = sigma_micro
    variables.sigma_micro_shape = sigma_micro_shape
    variables.sigma_micro_scale = sigma_micro_scale
    variables.sigma_back = sigma_back
    variables.sigma_back_shape = sigma_back_shape
    variables.sigma_back_scale = sigma_back_scale

    # Binding rates
    verbose ? println("--Binding rates") : nothing
    k_bind_shape = k_bind_shape .* ones(Float64, 3, 2)
    if k_bind === nothing
        # Set some bind rates
        bind_rate = .01/(dt*num_frames)
        unbind_rate = 100/(dt*num_frames)
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
    k_micro_shape = k_micro_shape .* ones(Float64, num_micro + 1, num_micro)
    if k_micro === nothing
        # Set some microphysical rates
        switch_rate = (.01 * num_max) / (dt * num_frames)
        bleach_rate = (.01 * num_max) / (dt * num_frames)
        blink_rate = (.01 * num_max) / (dt * num_frames)
        unblink_rate = (100 * num_max) / (dt * num_frames)
        k_micro = zeros(Float64, num_micro+1, num_micro)
        # Initial state probability
        # Photostate transition rates
        if num_micro == 2
            k_micro[end, :] .= [1, 0]
            k_micro[1, :] .= [-bleach_rate, bleach_rate]
        else
            k_micro[end, :] .= [0, fill(1/(num_micro-2), num_micro-2)..., 0]
            k_micro[1, :] .= [
                -unblink_rate, 
                fill(unblink_rate/(num_micro-2), num_micro-2)..., 
                0
            ]
            for k = 2:num_micro-1
                k_micro[k, :] .= [
                    blink_rate, 
                    fill(switch_rate/(num_micro-2), num_micro-2)..., 
                    bleach_rate
                ]
                k_micro[k, k] -= sum(k_micro[k, :])
            end
        end
    end
    if k_micro_scale === nothing
        k_micro_scale = copy(k_micro)
        k_micro_scale[1:end-1, :] ./= k_micro_shape[1:end-1, :]
    end
    variables.k_micro = k_micro
    variables.k_micro_shape = k_micro_shape
    variables.k_micro_scale = k_micro_scale

    # State partitions
    verbose ? println("--State partitions") : nothing
    partitions = calculate_partitions(num_max, num_micro)
    num_macro = size(partitions, 1)
    variables.partitions = partitions
    variables.num_macro = num_macro

    # Macrostate trajectory
    verbose ? println("--Macrostate trajectory") : nothing  
    if macrostates === nothing
        macrostates = num_macro * ones(Int, num_rois, num_frames)
    end
    variables.macrostates = macrostates

    # Number of bound fluorophores
    verbose ? println("--Number of bound fluorophores") : nothing
    num_bound = zeros(Int, num_rois)
    for r in 1:num_rois
        for n in 1:num_frames
            s = macrostates[r, n]
            num_bound[r] = maximum((num_bound[r], sum(partitions[s, 1:end-1])))
        end
    end
    variables.num_bound = num_bound

    # Find degenerate macrostate brightnesses
    verbose ? println("--Degenerate macrostates") : nothing
    mu_macro = partitions * mu_micro
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
    verbose ? println("Variables initialized.") : nothing

    # Return variables
    return variables
end # function initialize_variables

