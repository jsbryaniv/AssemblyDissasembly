
function sample_macrostates(data, variables; kwargs...)

    # Set up variables
    variables = SimpleNamespace(
        merge(Dict(variables), Dict([(string(key), val) for (key, val) in pairs(kwargs)]))
    )

    # Get constants
    dt = variables.dt
    partitions = variables.partitions
    degenerate_ids = variables.degenerate_ids
    mu_photo = variables.mu_photo
    mu_back = variables.mu_back
    sigma_photo = variables.sigma_photo
    sigma_back = variables.sigma_back
    k_photo = variables.k_photo
    num_rois = variables.num_rois
    num_frames = variables.num_frames
    num_photo = variables.num_photo
    num_macro = variables.num_macro
    num_unique = variables.num_unique
    background_times = variables.background_times

    # Calculate values
    eps = 1e-100 # for numerical stability
    threshold_prob = 1e-6
    not_background_states = findall(vec(sum(partitions[:, 1:end-1], dims=2)) .> 0)

    # Initialize variables
    macrostates = SharedArray{Int}(num_rois, num_frames)
    num_bound = SharedArray{Int}(num_rois)

    # Loop through each ROIs
    for r = 1:num_rois
        
        # Calculate ROI specific values
        mu_macro = partitions * mu_photo .+ mu_back[r]
        sigma_macro = partitions * sigma_photo .+ sigma_back[r]
        pi_macro = micro_rates_to_macro_transitions(
            dt, k_photo, partitions=partitions
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
            
            # Set background_times
            if background_times[r, n]
                lhood_n[not_background_states] .= - Inf
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
            M = sum(partitions[s, 1:end-1])
            M > num_bound[r] ? num_bound[r] = M : nothing
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
