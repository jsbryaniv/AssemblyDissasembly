
function sample_macrostates(data, variables; kwargs...)

    # Set up variables
    variables = SimpleNamespace(
        merge(Dict(variables), Dict([(string(key), val) for (key, val) in pairs(kwargs)]))
    )

    # Get constants
    dt = variables.dt
    partitions = variables.partitions
    degenerate_ids = variables.degenerate_ids
    mu_micro = variables.mu_micro
    mu_back = variables.mu_back
    sigma_micro = variables.sigma_micro
    sigma_back = variables.sigma_back
    k_micro = variables.k_micro
    k_bind = variables.k_bind
    concentration = variables.concentration
    laser_power = variables.laser_power
    num_rois = variables.num_rois
    num_frames = variables.num_frames
    num_micro = variables.num_micro
    num_macro = variables.num_macro
    num_unique = variables.num_unique

    # Calculate values
    eps = 1e-100 # for numerical stability
    threshold_prob = 1e-6

    # Initialize variables
    macrostates = SharedArray{Int}(num_rois, num_frames)
    num_bound = SharedArray{Int}(num_rois)

    # Loop through each ROIs
    # @sync @distributed for r = 1:num_rois
    for r = 1:num_rois
        
        # Calculate ROI specific values
        mu_macro = (partitions * mu_micro) .* laser_power[r] .+ mu_back[r]
        sigma_macro = (partitions * sigma_micro) .* laser_power[r] .+ sigma_back[r]
        pi_macro = micro_rates_to_macro_transitions(
            dt, k_micro, k_bind, C=concentration[r], W=laser_power[r], partitions=partitions
        )

        # Initialize log likelihood matrix
        lhood = spzeros(Float64, num_macro, num_frames)

        # Loop through time levels
        for n = 1:num_frames 

            # Create vector of likelihoods for time level n
            lhood_n = zeros(Float64, num_macro)

            # Create likelihood from only unique states
            for u = 1:num_unique
                
                # Find indexes and brightness
                idu = degenerate_ids[u]
                mu = mu_macro[idu[1]]
                sigma = sigma_macro[idu[1]]

                # Calculate likelihood
                lhood_n[idu] .= logpdf(Normal(mu, sigma), data[r, n])
            end
            
            # Softmax to get probabilities
            lhood_n = exp.(lhood_n .- maximum(lhood_n))  # softmax
            lhood_n ./= sum(lhood_n)                     # normalize

            # Exclude unprobable macrostates for memory usage
            idx = findall(lhood_n .> threshold_prob)

            # Fill likelihood matrix
            lhood[idx, n] .= lhood_n[idx]
        end

        # Forward filter
        forward = spzeros(num_macro, num_frames)
        forward_n = (lhood[:, 1] .+ eps) .* pi_macro[end, :]
        forward_n ./= sum(forward_n)
        idx = findall(forward_n .> threshold_prob)
        forward_n[idx] ./= sum(forward_n[idx])
        forward[idx, 1] .= forward_n[idx]
        for n = 2:num_frames
            forward_n = pi_macro[1:end-1, :]' * forward[:, n - 1]
            forward_n = (lhood[:, n] .+ eps) .* forward_n
            forward_n ./= sum(forward_n)
            idx = forward_n .> threshold_prob
            forward_n[idx] ./= sum(forward_n[idx])
            forward[idx, n] .= forward_n[idx]
        end

        # Backward sample
        s = rand(Categorical(forward[:, end]))
        macrostates[r, end] = s
        for n in num_frames-1:-1:1
            backward = pi_macro[1:end-1, s] .* forward[:, n]
            backward ./= sum(backward)
            s = rand(Categorical(backward))
            macrostates[r, n] = s
        end

        # Calculate number of bound states
        for n in 1:num_frames
            s = macrostates[r, n]
            num_bound[r] = maximum((num_bound[r], sum(partitions[s, 1:end-1])))
        end
    end

    # Convert back to arrays
    macrostates = convert(Array{Int}, macrostates)
    num_bound = convert(Array{Int}, num_bound)

    # Update variables
    variables.num_bound = num_bound
    variables.macrostates = macrostates

    # Return variables
    return variables
end # sample_macrostates

