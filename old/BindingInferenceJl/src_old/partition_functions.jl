
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


function micro_rates_to_macro_transitions(dt, k_micro, k_bind; C=1, W=1, partitions=1, index=nothing)
    """
    This function creates the transition matrix for the populations. It relies on 
    the assumption that only one transition can occur per time step.

    Let K be the number of microstates
    Let L be the number of Partitions

    Inputs:
        dt:
            Time step
        k_micro:
            (K+1 X K) transition rates between microstates. Last row is 
            initial state probability
        k_onoff:
            (2,) binding and unbinding rates
        C:
            Concentration of fluorophores
        L:
            Laser Power
        partitions:
            (L X K) array of all possible population combinations
            OR
            an integer for the number of particles
    Outputs
        pi_macro:
            (2*L+1 X 2*L) transition matrix between population combinations.
            Last row is initial state probability
    """

    # Set up partitions
    if isa(partitions, Number)
        num_micro = size(k_micro, 2)
        partitions = calculate_partitions(partitions, num_micro)
    end
    num_macro = size(partitions, 1)

    # Rescale rates
    if W != 1
        k_micro = copy(k_micro)
        for k in 1:size(k_micro, 2)
            k_micro[k, end] *= W
            k_micro[k, k] -= sum(k_micro[k, :])
        end
    end
    if C != 1
        k_bind = copy(k_bind)
        k_bind[2, :] .*= C
    end

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
                dt, k_micro, k_bind, partitions=partitions, index=(num_macro+1,i)
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
                findall(vec(sum(abs.(pop_diffs), dims=2) .== 2))...,  # only one +1 -1 transition occurred
            ]
            for j in allowed_transitions
                pi_transition = micro_rates_to_macro_transitions(
                    dt, k_micro, k_bind, partitions=partitions, index=(i,j)
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
            M = sum(partitions[j, 1:end-1])
            num_max = sum(partitions[1, :])
            pi_initial = (
                prod(k_bind[end, :] .^ [M, num_max-M]) 
                * multicoeff([M, num_max-M])
                * prod(k_micro[end, 1:end-1] .^ partitions[j, 1:end-1])
                * multicoeff(partitions[j, 1:end-1])
            )
            return pi_initial
        end

        # Calculate values
        M = sum(partitions[i, 1:end-1])
        num_max = sum(partitions[1, :])
        kon = k_bind[2, 1]
        koff = k_bind[1, 2]
        esc_rates = copy(k_micro[1:end-1, :])
        esc_rates[diagind(esc_rates)] .= 0
        esc_rates = sum(esc_rates, dims=2)  # total rate to leave each state
        total_rate = (
            (partitions[i, :]' * esc_rates)[1] 
            + (M * koff) 
            + kon*(M != num_max)
        )
        pi_self = exp(- dt * total_rate)
        
        # Check for self transition
        if i == j
            # Self transition
            return pi_self
        end

        # Find which transition occurred
        pop_diff = partitions[j, :] .- partitions[i, :]
        s_old = findall(pop_diff .== -1)[1]
        s_new = findall(pop_diff .== 1)[1]
        if pop_diff[end] == 0
            # Phototransition
            pi_transition = (
                (1 - pi_self) 
                * (partitions[i, s_old] * k_micro[s_old, s_new] / total_rate)
            )
        elseif pop_diff[end] == -1
            # Binding
            pi_transition = (
                (1 - pi_self) 
                * (kon / total_rate) 
                * (k_micro[end, s_new])
            )
        elseif pop_diff[end] == 1
            # Unbinding or photobleaching
            pi_transition = (
                (1 - pi_self) 
                * (partitions[i, s_old] * (koff + k_micro[s_old, s_new]) / total_rate)
            )
        end
        return pi_transition
    end

    # Return output
    return output
end # function micro_rates_to_macro_transitions 
