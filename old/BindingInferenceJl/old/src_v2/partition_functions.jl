
function multicoeff(k)::Float64
    # Calculates multinomial coefficient
    return exp(loggamma(sum(k) + 1) -sum(loggamma.(k .+ 1)))
end # function multicoeff


function calculate_partitions(n, b)::Array{Int64}
    """
    This function returns the number of partitions of n into b parts.
    For example, how to partition n balls into b bins, or n fluorophores
    into b microstates.

    Copied from algorithm created by karakfa on StackOverflow:
    https://stackoverflow.com/questions/37711817/...
    ...generate-all-possible-outcomes-of-k-balls-in-n-bins-sum-of-multinomial-catego

    :param n: The number to partition
    :param b: The number of parts
    :return: The number of partitions
    """

    # Check for trivial cases
    if b == 1
        partitions = zeros(Int, 1, 1)
        partitions[1, 1] = n
        return partitions
    elseif n == 0
        partitions = zeros(Int, 1, b)
        return partitions
    end

    # Calculate number of partitions
    num_partitions = binomial(n+b-1,n)

    # Intitialize partitions vector
    partitions = zeros(Int, num_partitions, b)

    # Loop through the n
    index = 1
    for i in n:-1:0

        # Find partitions for remainder
        temp = calculate_partitions(n-i, b-1)
        num_temp = size(temp, 1)

        # Fill x
        partitions[index:index+num_temp-1, 1] .= i
        partitions[index:index+num_temp-1, 2:end] .= temp

        # Reset index
        index = index + num_temp
    end

    # Return output
    return partitions
end # function calculate_partitions


function micro_rates_to_macro_transitions(dt, k_photo; partitions=nothing, num_max=nothing, index=nothing)

    # Set up constants
    num_photo = size(k_photo, 2)
    if partitions === nothing
        partitions = calculate_partitions(num_max, num_photo+1)[:, 1:end-1]
    else
        num_max =  maximum(sum(partitions, dims=2))
    end
    num_macro = size(partitions, 1)

    # If index is nothing construct full matrix
    if index === nothing
        
        # Initialize
        rows = Int[]
        cols = Int[]
        vals = Float64[]

        # Loop through all possible combinations of states
        for i in 1:num_macro
            # Set iniial probability
            pi_initial = micro_rates_to_macro_transitions(
                dt, k_photo, partitions=partitions, num_max=num_max, index=(num_macro+1,i)
            )
            if pi_initial > 0
                push!(rows, num_macro+1)
                push!(cols, i)
                push!(vals, pi_initial)
            end

            # Loop through allowed transitions
            pop_diffs = partitions .- partitions[i, :]'
            allowed_transitions = [
                # Self transition
                i,
                # Phototransitions
                findall(
                    vec(sum(pop_diffs, dims=2) .== 0)           # total pop is the same
                    .* vec(sum(abs.(pop_diffs), dims=2) .== 2)  # only one +1 -1 transition occurred
                )...,
            ]
            for j in allowed_transitions
                pi_transition = micro_rates_to_macro_transitions(
                    dt, k_photo, partitions=partitions, num_max=num_max, index=(i,j)
                )
                if pi_transition > 0
                    push!(rows, i)
                    push!(cols, j)
                    push!(vals, pi_transition)
                end
            end
        end

        # Convert to sparse matrix
        pi_macro = sparse(rows, cols, vals, num_macro + 1, num_macro)

        # Return transition matrix
        return pi_macro
    else

        # Get index
        i, j = index

        # Check for initial probability
        if i == num_macro + 1
            # Initial probability
            M = sum(partitions[j, :])
            pi_initial = (
                prod(k_photo[end, :] .^ partitions[j, :]) * multicoeff(partitions[j, :])
            )
            return pi_initial
        end

        # Calculate values
        esc_rates = copy(k_photo[1:end-1, :])
        esc_rates[diagind(esc_rates)] .= 0
        esc_rates = sum(esc_rates, dims=2)  # total rate to leave each state
        total_rate = (
            (partitions[i, :]' * esc_rates)[1]
        )
        pi_self = exp(- dt * total_rate)
        
        # Check for which transition occurred
        if i == j
            # Self transition
            return pi_self
        else
            # Find which transition occurred
            pop_diff = partitions[j, :] .- partitions[i, :]
            s_old = findall(pop_diff .== -1)[1]
            s_new = findall(pop_diff .== 1)[1]
            pi_transition = (
                (1 - pi_self) 
                * (partitions[i, s_old] * k_photo[s_old, s_new] / total_rate)
            )
            return pi_transition
        end
    end
end # function micro_rates_to_macro_transitions 
