
function sample_rates(data, variables; kwargs...)

    # Set up variables
    variables = SimpleNamespace(
        merge(Dict(variables), Dict([(string(key), val) for (key, val) in pairs(kwargs)]))
    )

    # Get variables
    dt = variables.dt
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
    num_max = variables.num_max
    num_rois = variables.num_rois
    num_data = variables.num_data
    num_macro = variables.num_macro
    num_photo = variables.num_photo
    alphap = variables.k_photo_prop_shape
    alphab = variables.k_bind_prop_shape

    # Initialize variables
    k_photo = copy(k_photo)
    k_bind = copy(k_bind)

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
    idp = findall(k_photo_scale[1:end-1, :] .> 0)
    idp0 = findall(k_photo_scale[end, :] .> 0)
    if k_bind_scale != 0
        idb = findall(k_bind_scale[1:end-1, :] .> 0)
        idb0 = findall(k_bind_scale[end, :] .> 0)
    end

    # Create probability function
    function probability(k_bind_, k_photo_)

        # Prior
        prob = (
            sum(logpdf.(Gamma.(k_photo_shape[idp], k_photo_scale[idp]), k_photo_[idp]))
        )
        if k_bind_scale != 0
            prob += sum(logpdf.(Gamma.(k_bind_shape[idb], k_bind_scale[idb]), k_bind_[idb]))
        end

        # Likelihood
        for r in 1:num_rois
            for (i, j) in Tuple.(findall(!iszero, macro_counts[r][1:end-1, :]))
                prob += macro_counts[r][i, j] .* log(
                    micro_rates_to_macro_transitions(
                        dt, k_photo, k_bind, C=concentration[r], W=laser_power[r], partitions=partitions, index=(i,j)
                    )
                )
            end
        end

        # Return probability
        return prob
    end

    # Sample phototransition rates
    k_photo[end, idp0] .= rand(Dirichlet(k_photo_scale[end, idp0] .+ photo_init_counts[idp0]))
    for i = 1:num_photo
        for j = 1:num_photo
            k_photo_scale[i, j] <= 0 ? continue : nothing
            # Sample multiple times using Metropolis Hastings
            for _ in 1:10
                k_photo_old = copy(k_photo)
                k_photo_new = copy(k_photo_old)
                k_photo_new[i, j] = rand(Gamma(alphap, k_photo_old[i,j] / alphap))
                k_photo_new[i, i] -= sum(k_photo_new[i, :])
                accept_prob = (
                    probability(k_bind, k_photo_new)
                    - probability(k_bind, k_photo_old)
                    + logpdf(Gamma(alphap, k_photo_new[i, j] / alphap), k_photo_old[i, j])
                    - logpdf(Gamma(alphap, k_photo_old[i, j] / alphap), k_photo_new[i, j])
                )
                if accept_prob > log(rand())
                    k_photo[:, :] .= k_photo_new[:, :]
                end
            end
        end
    end

    # Sample binding rates
    if k_bind_scale != 0
        k_bind[end, idb0] .= rand(Dirichlet(k_bind_scale[end, idb0] .+ bind_init_counts[idb0]))
        for i = 1:2
            for j = 1:2
                k_bind_scale[i, j] <= 0 ? continue : nothing
                # Sample multiple times using Metropolis Hastings
                for _ in 1:10
                    k_bind_old = copy(k_bind)
                    k_bind_new = copy(k_bind_old)
                    k_bind_new[i, j] = rand(Gamma(alphab, k_bind_old[i,j] / alphab))
                    k_bind_new[i, i] -= sum(k_bind_new[i, :])
                    accept_prob = (
                        probability(k_bind_new, k_photo)
                        - probability(k_bind_old, k_photo)
                        + logpdf(Gamma(alphab, k_bind_new[i, j] / alphab), k_bind_old[i, j])
                        - logpdf(Gamma(alphab, k_bind_old[i, j] / alphab), k_bind_new[i, j])
                    )
                    if accept_prob > log(rand())
                        k_bind[:, :] .= k_bind_new[:, :]
                    end
                end
            end
        end
    end

    # Update variables
    variables.k_photo = k_photo
    variables.k_bind = k_bind

    # Return variables
    return variables
end # function update_variables
