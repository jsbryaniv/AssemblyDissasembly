
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
    macrostates = variables.macrostates
    partitions = variables.partitions
    num_rois = variables.num_rois
    num_frames = variables.num_frames
    num_macro = variables.num_macro
    num_photo = variables.num_photo

    # Calculate variables
    ids = findall(mu_photo_mean .> 0)  # bright state IDs
    num_mus = length(ids) + num_rois

    # Reshape populations and add background
    pops = zeros(Float64, num_mus, num_rois * num_frames)
    dataflat = zeros(Float64, num_rois * num_frames)
    Sigma_inv = zeros(Float64, num_rois * num_frames)
    for r in 1:num_rois
        sigma_macro = partitions * sigma_photo .+ sigma_back[r]
        for n in 1:num_frames
            s = macrostates[r, n]
            pops[1:end-num_rois, n+num_frames*(r-1)] .= partitions[s, ids]
            pops[end-num_rois+r, n+num_frames*(r-1)] = 1
            dataflat[n+num_frames*(r-1)] = data[r, n]
            Sigma_inv[n+num_frames*(r-1)] = 1 / (sigma_macro[s] .^ 2)
        end
    end
    Sigma_inv = spdiagm(Sigma_inv)

    # Set up prior
    mu0 = [mu_photo_mean[ids]..., mu_back_mean...]
    Sigma0_inv = diagm(1 ./ ([mu_photo_std[ids]..., mu_back_std...].^2))

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

