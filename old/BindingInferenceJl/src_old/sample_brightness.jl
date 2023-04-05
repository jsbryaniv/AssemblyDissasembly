
function sample_brightness(data, variables; kwargs...)

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
    num_macro = variables.num_macro
    num_micro = variables.num_micro

    # Calculate variables
    ids = findall(mu_micro_mean .> 0)  # bright state IDs
    num_mus = length(ids) + num_rois

    # Reshape populations and add background
    pops = zeros(Float64, num_mus, num_rois * num_frames)
    dataflat = zeros(Float64, num_rois * num_frames)
    Sigma_inv = zeros(Float64, num_rois * num_frames)
    for r in 1:num_rois
        sigma_macro = (partitions * sigma_micro) .* laser_power[r] .+ sigma_back[r]
        for n in 1:num_frames
            s = macrostates[r, n]
            pops[1:end-num_rois, n+num_frames*(r-1)] .= partitions[s, ids] .* laser_power[r]
            pops[end-num_rois+r, n+num_frames*(r-1)] = 1
            dataflat[n+num_frames*(r-1)] = data[r, n]
            Sigma_inv[n+num_frames*(r-1)] = 1 / (sigma_macro[s] .^ 2)
        end
    end
    Sigma_inv = spdiagm(Sigma_inv)

    # Set up prior
    mu0 = [mu_micro_mean[ids]..., mu_back_mean...]
    Sigma0_inv = diagm(1 ./ ([mu_micro_std[ids]..., mu_back_std...].^2))

    # Sample brightnesses
    cov = Sigma0_inv + pops * Sigma_inv * pops'
    cov = inv(cov) # + .01 * maximum(cov) * Matrix(I, num_mus, num_mus))
    cov = (cov + cov')/2
    mu = cov * (Sigma0_inv * mu0 + pops * Sigma_inv * dataflat)
    q = rand(MvNormal(vec(mu), cov))
    mu_micro[ids] = q[1:length(ids)]
    mu_back = q[length(ids)+1:end]

    # Update variables
    variables.mu_micro = mu_micro
    variables.mu_back = mu_back

    # Return variables
    return variables
end # sample_brightness
