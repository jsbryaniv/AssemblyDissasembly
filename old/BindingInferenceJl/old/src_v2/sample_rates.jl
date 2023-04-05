
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
    macrostates = variables.macrostates
    partitions = variables.partitions
    num_max = variables.num_max
    num_rois = variables.num_rois
    num_frames = variables.num_frames
    num_macro = variables.num_macro
    num_photo = variables.num_photo
    alphap = variables.k_photo_prop_shape

    # Initialize variables
    k_photo = copy(k_photo)

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
    idp = findall(k_photo_scale[1:end-1, :] .> 0)
    idp0 = findall(k_photo_scale[end, :] .> 0)

    # Create probability function
    function probability(k_photo_)

        # Prior
        prob = (
            sum(logpdf.(Gamma.(k_photo_shape[idp], k_photo_scale[idp]), k_photo_[idp]))
        )

        # Likelihood
        for r in 1:num_rois
            for (i, j) in Tuple.(findall(!iszero, macro_counts[r]))
                prob += macro_counts[r][i, j] .* log(
                    micro_rates_to_macro_transitions(
                        dt, k_photo, partitions=partitions, num_max=num_max, index=(i,j)
                    )
                )
            end
        end

        # Return probability
        return prob
    end

    # Sample phototransition rates
    # k_photo[end, idp0] .= rand(Dirichlet(k_photo_scale[end, idp0] .+ macro_counts[end, idp0]))
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
                    probability(k_photo_new)
                    - probability(k_photo_old)
                    + logpdf(Gamma(alphap, k_photo_new[i, j] / alphap), k_photo_old[i, j])
                    - logpdf(Gamma(alphap, k_photo_old[i, j] / alphap), k_photo_new[i, j])
                )
                if accept_prob > log(rand())
                    k_photo[:, :] .= k_photo_new[:, :]
                end
            end
        end
    end

    # Update variables
    variables.k_photo = k_photo

    # Return variables
    return variables
end # function update_variables
