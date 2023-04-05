
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
    macro_counts_tot = sum(macro_counts)
    
    # Count initial states
    bind_init_counts = zeros(Int, 2)  # bindstate initial state counts
    photo_init_counts = zeros(Int, num_micro)  # photostate initial state counts
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
    ids = findall(mu_micro .!= 0)
    idp = findall(k_micro_scale[1:end-1, :] .> 0)
    idp0 = findall(k_micro_scale[end, :] .> 0)
    if k_bind_scale != 0
        idb = findall(k_bind_scale[1:end-1, :] .> 0)
        idb0 = findall(k_bind_scale[end, :] .> 0)
    end

    # Calculate prior
    verbose ? println("Calculating prior") : nothing
    prior = (
        sum(logpdf.(Gamma.(k_micro_shape[idp], k_micro_scale[idp]), k_micro[idp]))
        # + logpdf(Dirichlet(k_micro[end, idp0]), k_micro_scale[end, idp0])
    )
    if k_bind_scale !== 0
        prior += (
            logpdf(Dirichlet(k_bind[end, idb0]), k_bind_scale[end, idb0])
            # + sum(logpdf.(Gamma.(k_bind_shape[idb], k_bind_scale[idb]), k_bind[idb]))
        )
    end
    prior += sum(logpdf.(Normal.(mu_back_mean, mu_back_std), mu_back))
    prior += sum(logpdf.(Normal.(mu_micro_mean[ids], mu_micro_std[ids]), mu_micro[ids]))
    prior += sum(logpdf.(Gamma.(sigma_back_shape, sigma_back_scale), sigma_back))
    prior += sum(logpdf.(Gamma.(sigma_micro_shape[ids], sigma_micro_scale[ids]), sigma_micro[ids]))
    

    # Calculate kinetic part
    verbose ? println("Calculating dynamics part") : nothing
    kinetic = 0
    for r in 1:num_rois
        for (i, j) in Tuple.(findall(!iszero, macro_counts[r][1:end-1, :]))
            kinetic += macro_counts[r][i, j] .* log(
                micro_rates_to_macro_transitions(
                    dt, k_micro, k_bind, C=concentration[r], W=laser_power[r], partitions=partitions, index=(i,j)
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
        mu_macro = (partitions * mu_micro) .* laser_power[r] .+ mu_back[r]
        sigma_macro = (partitions * sigma_micro) .* laser_power[r] .+ sigma_back[r]
        
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
