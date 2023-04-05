
function sample_states(data, variables; kwargs...)

    # Set up variables
    variables = SimpleNamespace(
        merge(Dict(variables), Dict([(string(key), val) for (key, val) in pairs(kwargs)]))
    )

    # Get constants
    P = variables.P
    pi = variables.pi
    gain = variables.gain
    states = variables.states
    mu_flor = variables.mu_flor
    mu_back = variables.mu_back
    gain_shape = variables.gain_shape
    gain_scale = variables.gain_scale
    mu_flor_shape = variables.mu_flor_shape
    mu_flor_scale = variables.mu_flor_scale
    mu_back_shape = variables.mu_back_shape
    mu_back_scale = variables.mu_back_scale
    pi_concentration = variables.pi_concentration
    num_frames = variables.num_frames
    num_states = variables.num_states
    seed = variables.seed
    mu_flor_prop_shape = variables.mu_flor_prop_shape
    mu_back_prop_shape = variables.mu_back_prop_shape
        
    # Initialize log likelihood matrix
    lhood = zeros(Float64, num_states, num_frames)
    scale = mu_flor * ((1:num_states) .- 1) .+ mu_back
    for n = 1:num_frames
        lhood[:, n] .= logpdf.(Gamma.(gain, scale), data[n])
        lhood[:, n] .-= maximum(lhood[:, n])
        lhood[:, n] .= exp.(lhood[:, n])
        lhood[:, n] ./= sum(lhood[:, n])
    end
    lhood .+= 1e-100

    # Forward filter
    forward = zeros(num_states, num_frames)
    forward[:, 1] .= lhood[:, 1] .* pi[end, :]
    forward[:, 1] ./= sum(forward[:, 1])
    for n = 2:num_frames
        forward[:, n] .= lhood[:, n] .* (pi[1:end-1, :]' * forward[:, n - 1])
        forward[:, n] ./= sum(forward[:, n])
    end
    forward .+= 1e-100

    # Backward sample
    s = rand(Categorical(forward[:, end]))
    states[end] = s
    for n in num_frames-1:-1:1
        backward = pi[1:end-1, s] .* forward[:, n]
        backward ./= sum(backward)
        s = rand(Categorical(backward))
        states[n] = s
    end

    # Update variables
    variables.states = states

    # Return variables
    return variables
end

