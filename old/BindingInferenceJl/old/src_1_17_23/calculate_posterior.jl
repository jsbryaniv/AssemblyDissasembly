
function calculate_posterior(data, variables; verbose=false, kwargs...)

    # Set up variables
    verbose ? println("Setting up variables") : nothing
    variables = SimpleNamespace(
        merge(Dict(variables), Dict([(string(key), val) for (key, val) in pairs(kwargs)]))
    )

    # Get variables
    verbose ? println("Extracting variables") : nothing
    dt = variables.dt
    gain = variables.gain
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
    num_max = variables.num_max
    cameramodel = variables.cameramodel

    # Count transitions
    macro_counts = [spzeros(Int, num_macro+1, num_macro) for _ in 1:num_rois]
    for r = 1:num_rois
        s_old = num_macro + 1
        for n = 2:num_data
            s_new = macrostates[r, n]
            macro_counts[r][s_old, s_new] += 1
            s_old = s_new
        end
    end
    macro_counts_tot = sum(macro_counts)
    
    # Count initial states
    bind_init_counts = zeros(Int, 2)  # bindstate initial state counts
    photo_init_counts = zeros(Int, num_photo)  # photostate initial state counts
    for i in 1:num_macro
        pops = partitions[i, :]
        bind_init_counts .+= macro_counts_tot[end, i] .* [sum(pops), num_max - sum(pops)]
        photo_init_counts .+= macro_counts_tot[end, i] .* pops 
        for j in findall(macro_counts_tot[i, :] .> 0)
            pop_diff = partitions[j, :] - pops
            if (sum(pop_diff) == 1) && (sum(pop_diff .== 1) == 1)
                photo_init_counts .+= macro_counts_tot[i, j] .* pop_diff
            end
        end
    end

    # Find nonzeros
    verbose ? println("Finding nonzeros") : nothing
    ids = findall(mu_photo .!= 0)
    idp = findall(k_photo_scale[1:end-1, :] .> 0)
    idp0 = findall(k_photo_scale[end, :] .> 0)
    if k_bind_scale != 0
        idb = findall(k_bind_scale[1:end-1, :] .> 0)
        idb0 = findall(k_bind_scale[end, :] .> 0)
    end

    # Calculate prior
    verbose ? println("Calculating prior") : nothing
    prior = (
        sum(logpdf.(Gamma.(k_photo_shape[idp], k_photo_scale[idp]), k_photo[idp]))
        + logpdf(Dirichlet(k_photo[end, idp0]), k_photo_scale[end, idp0])
    )
    if k_bind_scale !== 0
        prior += (
            logpdf(Dirichlet(k_bind[end, idb0]), k_bind_scale[end, idb0])
            + sum(logpdf.(Gamma.(k_bind_shape[idb], k_bind_scale[idb]), k_bind[idb]))
        )
    end
    if cameramodel == "emccd"
        prior += (
            sum(logpdf.(Gamma.(mu_back_shape, mu_back_scale), mu_back))
            + sum(logpdf.(Gamma.(mu_photo_shape[ids], mu_photo_scale[ids]), mu_photo[ids]))
        )
    elseif cameramodel == "scmos"
        prior += (
            sum(logpdf.(Normal.(mu_back_mean, mu_back_vars), mu_back))
            + sum(logpdf.(Normal.(mu_photo_mean[ids], mu_photo_vars[ids]), mu_photo[ids]))
            + sum(logpdf.(Gamma.(sigma_back_shape, sigma_back_scale), sigma_back))
            + sum(logpdf.(Gamma.(sigma_photo_shape[ids], sigma_photo_scale[ids]), sigma_photo[ids]))
        )
    end

    # Calculate kinetic part
    verbose ? println("Calculating dynamics part") : nothing
    kinetic = 0
    for r in 1:num_rois
        for (i, j) in Tuple.(findall(!iszero, macro_counts[r][1:end-1, :]))
            kinetic += macro_counts[r][i, j] .* log(
                micro_rates_to_macro_transitions(
                    dt, k_photo, k_bind, C=concentration[r], W=laser_power[r], partitions=partitions, index=(i,j)
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
        mu_macro = (partitions * mu_photo) .* laser_power[r] .+ mu_back[r]
        if cameramodel === "scmos"
            sigma_macro = (partitions * sigma_photo) .* laser_power[r] .+ sigma_back[r]
        end
        
        # Loop through time levels
        for n = 1:num_data
            s = macrostates[r, n]
            if cameramodel == "emccd"
                lhood += logpdf(Gamma(.5*mu_macro[s], 2*gain), data[r, n])
            elseif cameramodel == "scmos"
                lhood += logpdf(Normal(mu_macro[s], sigma_macro[s]), data[r, n])
            end
        end
    end

    # Calculate posterior
    verbose ? println("Calculating posterior") : nothing
    posterior = prior + lhood + kinetic

    # Return posterior
    return posterior
end # calculate_posterior
