
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
    macrostates = variables.macrostates
    mu_back = variables.mu_back
    mu_back_mean = variables.mu_back_mean
    mu_back_std = variables.mu_back_std
    mu_photo = variables.mu_photo
    mu_photo_mean = variables.mu_photo_mean
    mu_photo_std = variables.mu_photo_std
    sigma_photo = variables.sigma_photo
    sigma_photo_shape = variables.sigma_photo_shape
    sigma_photo_scale = variables.sigma_photo_scale
    sigma_back = variables.sigma_back
    sigma_back_shape = variables.sigma_back_shape
    sigma_back_scale = variables.sigma_back_scale
    k_photo = variables.k_photo
    k_photo_shape = variables.k_photo_shape
    k_photo_scale = variables.k_photo_scale
    num_rois = variables.num_rois
    num_frames = variables.num_frames
    num_photo = variables.num_photo
    num_macro = variables.num_macro
    num_max = variables.num_max
    seed = variables.seed
    flor_brightness_guess = variables.flor_brightness_guess
    background_times = variables.background_times

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

    # Background times
    if background_times === nothing
        background_times = zeros(Bool, num_rois, num_frames)
    end
    variables.background_times = background_times

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
    if mu_photo === nothing
        if flor_brightness_guess === nothing
            flor_brightness_guess = .5*mean(maximum(data, dims=2) - mu_back)
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
    mu_photo = mu_photo .* ones(Float64, num_photo)
    if mu_photo_mean === nothing
        mu_photo_mean = mu_photo .* ones(Float64, num_photo)
    end
    if mu_photo_std === nothing
        mu_photo_std = mu_photo .* ones(Float64, num_photo)
    end
    variables.mu_photo = mu_photo
    variables.mu_photo_mean = mu_photo_mean
    variables.mu_photo_std = mu_photo_std

    # Brightness variance
    verbose ? println("--Brightness variance") : nothing
    if sigma_back === nothing
        sigma_back = zeros(Float64, num_rois)
        for r in 1:num_rois
            if any(background_times[r, :] .> 0)
                # Estimate background from background times if specified
                sigma_back[r] = std(data[r, findall(background_times[r, :])])
            else
                # Estimate background as the dimmest 10% of the data for each ROI
                sigma_back[r] = .5*std(data[r, :])
            end
        end
    end
    sigma_back = sigma_back .* ones(Float64, num_rois)
    sigma_back_shape = sigma_back_shape .* ones(Float64, num_rois)
    if sigma_back_scale === nothing
        sigma_back_scale = sigma_back ./ sigma_back_shape
    end
    if sigma_photo === nothing
        sigma_photo = .5*std(data)
    end
    sigma_photo = sigma_photo .* (mu_photo .> 0)
    sigma_photo_shape = sigma_photo_shape .* ones(Float64, num_photo)
    if sigma_photo_scale === nothing
        sigma_photo_scale = sigma_photo ./ sigma_photo_shape
    end
    variables.sigma_photo = sigma_photo
    variables.sigma_photo_shape = sigma_photo_shape
    variables.sigma_photo_scale = sigma_photo_scale
    variables.sigma_back = sigma_back
    variables.sigma_back_shape = sigma_back_shape
    variables.sigma_back_scale = sigma_back_scale

    # Photostate transition rates
    verbose ? println("--Photostate transition rates") : nothing
    k_photo_shape = k_photo_shape .* ones(Float64, num_photo + 1, num_photo)
    if k_photo === nothing
        k_photo = 10 * ones(Float64, num_photo + 1, num_photo) / dt / num_frames
        for i in 1:num_photo
            k_photo[i, i] -= sum(k_photo[i, :])
        end
        k_photo[end, :] /= sum(k_photo[end, :])
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
    partitions = calculate_partitions(num_max, num_photo)
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

