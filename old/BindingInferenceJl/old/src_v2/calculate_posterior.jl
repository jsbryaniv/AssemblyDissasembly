
function calculate_posterior(data, variables; verbose=false, kwargs...)

    # Set up variables
    verbose ? println("Setting up variables") : nothing
    variables = SimpleNamespace(
        merge(Dict(variables), Dict([(string(key), val) for (key, val) in pairs(kwargs)]))
    )

    # Get variables
    verbose ? println("Extracting variables") : nothing
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
    num_max = variables.num_max

    # Count transitions
    macro_counts = [spzeros(Int, num_macro+1, num_macro) for _ in 1:num_rois]
    for r = 1:num_rois
        s_old = num_macro + 1
        for n = 2:num_frames
            s_new = macrostates[r, n]
            macro_counts[r][s_old, s_new] += 1
            s_old = s_new
        end
    end

    # Find nonzeros
    verbose ? println("Finding nonzeros") : nothing
    ids = findall(mu_photo .!= 0)
    idp = findall(k_photo_scale[1:end-1, :] .> 0)
    idp0 = findall(k_photo_scale[end, :] .> 0)

    # Calculate prior
    verbose ? println("Calculating prior") : nothing
    prior = (
        sum(logpdf.(Gamma.(k_photo_shape[idp], k_photo_scale[idp]), k_photo[idp]))
        + logpdf(Dirichlet(k_photo[end, idp0]), k_photo_scale[end, idp0])
    )
    prior += (
        sum(logpdf.(Normal.(mu_back_mean, mu_back_std), mu_back))
        + sum(logpdf.(Normal.(mu_photo_mean[ids], mu_photo_std[ids]), mu_photo[ids]))
        + sum(logpdf.(Gamma.(sigma_back_shape, sigma_back_scale), sigma_back))
        + sum(logpdf.(Gamma.(sigma_photo_shape[ids], sigma_photo_scale[ids]), sigma_photo[ids]))
    )

    # Calculate kinetic part
    verbose ? println("Calculating dynamics part") : nothing
    kinetic = 0
    for r in 1:num_rois
        for (i, j) in Tuple.(findall(!iszero, macro_counts[r][1:end-1, :]))
            kinetic += macro_counts[r][i, j] .* log(
                micro_rates_to_macro_transitions(
                    dt, k_photo, partitions=partitions, num_max=num_max, index=(i,j)
                )
            )
        end
    end

    # Calculate likelihood
    verbose ? println("Calculating likelihood") : nothing
    lhood = 0
    for r = 1:num_rois
        verbose ? println("--ROI $(r)") : nothing

        # Get macrostate features
        mu_macro = partitions * mu_photo .+ mu_back[r]
        sigma_macro = partitions * sigma_photo .+ sigma_back[r]
        
        # Loop through time levels
        for n = 1:num_frames
            s = macrostates[r, n]
            lhood += logpdf(Normal(mu_macro[s], sigma_macro[s]), data[r, n])
        end
    end

    # Calculate posterior
    verbose ? println("Calculating posterior") : nothing
    posterior = prior + lhood + kinetic

    # Return posterior
    return posterior
end # calculate_posterior
